import { CohereClient } from "cohere-ai";

export interface Env {
	API_TOKEN: string;
	PAGE_METADATA: D1Database;
	KNOWLEDGE_BUCKET: R2Bucket;
	COHERE_API_KEY: string;
	COHERE_EMBEDDING_MODEL: string;
	KNOWLEDGE_INDEX: VectorizeIndex;
}

interface RequestBody {
	url: string;
}

export default {
	async fetch(request, env, ctx): Promise<Response> {
		// Check if the request is authorized
		const apiKey = request.headers.get("Authorization")?.replace("Bearer ", "");
		if (apiKey !== env.API_TOKEN) {
			console.log({ "message": "Unauthorized request", "APIKey": apiKey, "ExpectedAPIKey": env.API_TOKEN });
			return Response.json({"message": "Unauthorized", "status": "failed"}, { status: 401 });
		}

		// Check if the request is POST
		if (request.method !== "POST") {
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

		const targetUrl = new URL(reqUrl);
		const targetUrlString = targetUrl.toString();

		let r2Key: string;
		let docId: string;
		// Get HTML location from D1 PageMetadata
		try {
			const pageMetadata = await env.PAGE_METADATA.prepare("SELECT id, url, r2_path FROM PageMetadata WHERE url = ?")
				.bind(targetUrlString)
				.first();

			if (!pageMetadata) {
				console.log({ "message": "URL is not in the database", "URL": targetUrlString });
				return Response.json({"message": "URL is not in the database", "status": "failed"}, { status: 404 });
			}

			docId = String(pageMetadata.id);
			r2Key = String(pageMetadata.r2_path);
			console.log({ "message": "Fetched URL metadata from PageMetadata", "URL": targetUrlString, "R2Path": r2Key });
		} catch (e: any) {
			console.log({ "message": "Failed to query from D1 PageMetadata", "URL": targetUrlString, "Error": e.message });
			console.error(e);
			return Response.json({"message": "Failed to get URL metadata", "status": "failed"}, { status: 500 });
		}

		let markdown: string;
		// Get markdown from R2
		try {
			const r2Obj = await env.KNOWLEDGE_BUCKET.get(r2Key);
			if (!r2Obj) {
				console.log({ "message": "Markdown not found from R2", "R2Path": r2Key });
				return Response.json({"message": "Markdown not found from R2", "status": "failed"}, { status: 404 });
			}

			markdown = await r2Obj.text();
			console.log({ "message": "Fetched markdown from R2", "URL": targetUrlString, "R2Path": r2Key });
		} catch (e: any) {
			console.log({ "message": "Failed to get markdown from R2", "URL": targetUrlString, "R2Path": r2Key, "Error": e.message });
			console.error(e);
			return Response.json({"message": "Failed to get markdown", "status": "failed"}, { status: 500 });
		}

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
				texts: [markdown],
			})

			const embeddings = embed.embeddings['float'];

			billedUnits = embed.meta?.billedUnits;
			warnings = embed.meta?.warnings || [];

			if (!embeddings) {
				console.log({ 
					"message": "Failed to generate embeddings", 
					"URL": targetUrlString, 
					"id": embed.id, 
					"metadata": embed.meta
				});
				return Response.json({"message": "Failed to generate embeddings", "status": "failed"}, { status: 500 });
			} else {
				console.log({ 
					"message": "Generated embeddings", 
					"URL": targetUrlString, 
					"id": embed.id, 
					"billedUnits": billedUnits,
					"warnings": warnings,
					"metadata": embed.meta
				});

				vectors.push({id: `${docId}:0`, values: embeddings[0], namespace: "scrapped-markdown", metadata: {
					"url": targetUrlString,
					"markdown_r2_key": r2Key
				}});	
			}
		} catch (e: any) {
			console.log({ "message": "Failed to generate embeddings", "URL": targetUrlString, "Error": e.message });
			console.error(e);
			return Response.json({"message": "Failed to generate embeddings", "status": "failed"}, { status: 500 });
		}

		try {
			const upsertResult = await env.KNOWLEDGE_INDEX.upsert(vectors);
			console.log({ "message": "Upserted vectors", "id": docId, "URL": targetUrlString, "Result": upsertResult });
		} catch (e: any) {
			console.log({ "message": "Failed to upsert vectors", "id": docId, "URL": targetUrlString, "Error": e.message });
			console.error(e);
			return Response.json({"message": "Failed to upsert vectors", "status": "failed"}, { status: 500 });
		}

		// Update the PageMetadata with the embedding creation time
		try {
			await env.PAGE_METADATA.prepare("UPDATE PageMetadata SET embedding_created_at = CURRENT_TIMESTAMP WHERE id = ?")
				.bind(docId)
				.run();
		} catch (e: any) {
			console.log({ "message": "Failed to update PageMetadata", "id": docId, "Error": e.message });
			console.error(e);
		}

		return Response.json({"message": "Successfully upserted vectors", "status": "success", "billedUnits": billedUnits, "warnings": warnings}, { status: 200 });
	},
} satisfies ExportedHandler<Env>;
