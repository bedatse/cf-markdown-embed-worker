import { CohereClient } from "cohere-ai";

export interface Env {
	API_TOKEN: string;
	PAGE_METADATA: D1Database;
	KNOWLEDGE_BUCKET: R2Bucket;
	COHERE_API_KEY: string;
	COHERE_EMBEDDING_MODEL: string;
	KNOWLEDGE_INDEX: VectorizeIndex;
	GENERATE_EMBEDDING_QUEUE: Queue;
}

export interface RequestBody {
	url: string;
	namespace: string;
}

export interface MarkdownWithMetadata {
	markdown: string;
	url: string;
	docId: string;
	r2Key: string;
	namespace: string;
}

const DEFAULT_NAMESPACE = "markdown-rag";

/**
 * Get the page metadata from the D1 database
 * @param url - The URL to get the metadata for
 * @param env - The environment bindings
 * @returns The document ID and the R2 key
 */
export async function getPageMetadata(url:string, env: Env) {
	// Get HTML location from D1 PageMetadata
	try {
		const pageMetadata = await env.PAGE_METADATA.prepare("SELECT id, r2_path FROM PageMetadata WHERE url = ?")
			.bind(url)
			.first();

		if (!pageMetadata) {
			console.log({ "message": "URL is not in the database", "URL": url });
			return {docId: null, r2Key: null};
		}

		const docId = String(pageMetadata.id);
		const r2Key = String(pageMetadata.r2_path);
		console.log({ "message": "Fetched URL metadata from PageMetadata", "URL": url, "R2Path": r2Key });

		return {docId, r2Key}
	} catch (e: any) {
		console.log({ "message": "Failed to query from D1 PageMetadata", "URL": url, "Error": e.message });
		console.error(e);
		throw new Error("Failed to query from D1 PageMetadata");
	}
}

/**
 * Get the Markdown from the R2 bucket
 * @param r2Key - The R2 key to get the markdown from
 * @param env - The environment bindings
 * @returns The Markdown
 */
export async function getMarkdown(r2Key: string, env: Env) {
	// Get HTML from R2
	try {
		const r2Obj = await env.KNOWLEDGE_BUCKET.get(r2Key);
		if (!r2Obj) {
			console.log({ "message": "Markdown not found from R2", "R2Path": r2Key });
			return null;
		}

		const html = await r2Obj.text();
		console.log({ "message": "Fetched markdown from R2", "R2Path": r2Key });

		return html;
	} catch (e: any) {
		console.log({ "message": "Failed to get markdown from R2", "R2Path": r2Key, "Error": e.message });
		console.error(e);
		throw new Error("Failed to get markdown from R2")
	}
}

export async function externalEmbeddingAPI(markdownWithMetadata: MarkdownWithMetadata[], env: Env) {
	// Generate embeddings
	let vectors: VectorizeVector[] = [];
	let billedUnits: any = {};
	let warnings: string[];

	try {
		const cohere = new CohereClient({
			token: env.COHERE_API_KEY,
		});

		const embed = await cohere.v2.embed({
			model: env.COHERE_EMBEDDING_MODEL,
			inputType: "search_document",
			embeddingTypes: ['float'],
			texts: markdownWithMetadata.map((item) => item.markdown),
		})

		const embeddings = embed.embeddings['float'];

		billedUnits = embed.meta?.billedUnits;
		warnings = embed.meta?.warnings || [];

		if (!embeddings) {
			console.log({ 
				"message": "Failed to generate embeddings", 
				"id": embed.id, 
				"metadata": embed.meta
			});
			throw new Error("Failed to generate embeddings")
		} else {
			console.log({ 
				"message": "Generated embeddings", 
				"id": embed.id, 
				"billedUnits": billedUnits,
				"warnings": warnings,
				"metadata": embed.meta
			});

			for (let i = 0; i < markdownWithMetadata.length; i++) {
				vectors.push({
					id: `${markdownWithMetadata[i].docId}:0`, 
					values: embeddings[i], 
					namespace: markdownWithMetadata[i].namespace,
					metadata: {
						"url": markdownWithMetadata[i].url,
						"doc_id": markdownWithMetadata[i].docId,
						"r2_key": markdownWithMetadata[i].r2Key
					}
				});	
			}

			return {vectors, billedUnits, warnings};
		}
	} catch (e: any) {
		console.log({ "message": "Failed to generate embeddings", "Error": e.message });
		console.error(e);
		throw new Error("Failed to generate embeddings")
	}
}

export async function upsertVectors(vectors: VectorizeVector[], env: Env) {
	try {
		const upsertResult = await env.KNOWLEDGE_INDEX.upsert(vectors);
		console.log({ "message": "Upserted vectors", "Result": upsertResult });
	} catch (e: any) {
		console.log({ "message": "Failed to upsert vectors", "Error": e.message });
		console.error(e);
		throw new Error("Failed to upsert vectors")
	}
}

export async function updatePageMetadata(docId: string, env: Env) {
	try {
		await env.PAGE_METADATA.prepare("UPDATE PageMetadata SET embedding_created_at = CURRENT_TIMESTAMP WHERE id = ?")
			.bind(docId)
			.run();
	} catch (e: any) {
		console.log({ "message": "Failed to update PageMetadata", "id": docId, "Error": e.message });
		console.error(e);
	}
}

/**
 * Generate the markdown embeddings
 * @param body - The request body
 * @param env - The environment bindings
 * @returns The response
 */
export async function generateMarkdownEmbeddings(requests: RequestBody[], env: Env) {
	try {
		console.log({ "message": "Generating markdown embeddings", "NrRequests": requests.length });

		const markdownWithMetadata: MarkdownWithMetadata[] = [];
		for (const request of requests) {
			const {docId, r2Key} = await getPageMetadata(request.url, env);
			if (!docId || !r2Key) {
				console.log({ "message": "URL is not in the database", "URL": request.url });
				continue;
			}
			const markdown = await getMarkdown(r2Key, env);
			if (!markdown) {
				console.log({ "message": "Markdown not found from R2", "R2Path": r2Key });
				continue;
			}

			markdownWithMetadata.push({markdown, url: request.url, docId, r2Key, namespace: request.namespace});
		}

		if (markdownWithMetadata.length === 0) {
			console.log({ "message": "No valid markdown found" });
			return {"message": "No valid markdown found", "status": "hardfail", "code": 404};
		}

		console.log({ "message": "Calling external embedding API", "NrMarkdownWithMetadata": markdownWithMetadata.length });
		const {vectors, billedUnits, warnings} = await externalEmbeddingAPI(markdownWithMetadata, env);

		console.log({ "message": "Upserting vectors", "NrVectors": vectors.length });
		await upsertVectors(vectors, env);

		for (const vector of vectors) {
			if (vector.metadata?.doc_id) {
				await updatePageMetadata(String(vector.metadata?.doc_id), env);
			}
		}

		console.log({ "message": "Successfully upserted vectors", "billedUnits": billedUnits, "warnings": warnings });
		return {"message": "Successfully upserted vectors", "status": "success", "billedUnits": billedUnits, "warnings": warnings};
	} catch (e: any) {
		return {"message": e.message, "status": "softfail", "code": 500};
	}
}

export default {
	async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
		// Check if the request is authorized
		const apiKey = request.headers.get("Authorization")?.replace("Bearer ", "");
		if (apiKey !== env.API_TOKEN) {
			console.log({ "message": "Unauthorized request", "APIKey": apiKey, "ExpectedAPIKey": env.API_TOKEN });
			return Response.json({"message": "Unauthorized", "status": "failed"}, { status: 401 });
		}

		// Check if the request is POST
		if (request.method !== "POST" && request.method !== "PUT") {
			console.log({ "message": "Invalid request method", "Method": request.method });
			return Response.json({"message": "Invalid request method", "status": "failed"}, { status: 405 });
		}
		
		const body: RequestBody = await request.json();

		const reqUrl = body?.url;
		// Check if the URL is provided
		if (!reqUrl) {
			console.log({ "message": "URL parameter is missing", "URL": reqUrl });
			return Response.json({"message": "URL is required", "status": "failed"}, { status: 400 });
		} 

		const reqNamespace = body?.namespace || DEFAULT_NAMESPACE;

		switch (request.method) {
			case "POST":
				const result = await generateMarkdownEmbeddings([body], env);
				return Response.json({"message": result.message, "status": result.status, "billedUnits": result.billedUnits, "warnings": result.warnings}, { status: result.code });
			case "PUT":
				try {
					await env.GENERATE_EMBEDDING_QUEUE.send(body);
					return Response.json({"message": "Request Accepted", "status": "success", "request": body}, { status: 202 });
				} catch (e: any) {
					console.log({ "message": "Failed to send message to queue", "Error": e.message });
					console.error(e);
					return Response.json({"message": "Failed to send message to queue", "status": "failed"}, { status: 500 });
				}
		}
	},

	async queue(batch: MessageBatch, env: Env): Promise<void> {
		console.log({ "message": "Consuming queue", "BatchSize": batch.messages.length });

		const requests: RequestBody[] = batch.messages.map((message) => message.body as RequestBody);

		const result = await generateMarkdownEmbeddings(requests, env);

		switch (result.status) {
			case "hardfail":
				console.log({ "message": "Failed to generate embeddings, will not retry", "Result": result });
				batch.ackAll();
				break;
			case "softfail":
				console.log({ "message": "Failed to generate embeddings, will retry", "Result": result });
				batch.retryAll();
				break;
			case "success":
				console.log({ "message": "Successfully generated embeddings", "Result": result });
				batch.ackAll();
				break;
		}
	}
} satisfies ExportedHandler<Env>;
