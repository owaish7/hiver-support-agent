# Hiver Live Support — Vercel

This directory contains the live support copilot and historical evaluation explorer. Set the Vercel project's **Root Directory to `web`**, use the **Other** framework preset, and use the tracked `vercel.json` settings. Node 22 is required.

The website and live API execute on Vercel. There is no proxy or runtime dependency on the previous hosting service. Fresh support messages use Gemini, BM25 retrieval over demo or visitor-supplied policies, structured output/citation validation, conservative handoffs, and a local review queue. The original historical evaluation remains clearly labelled as a different model/retrieval pipeline.

## Setup

1. Import the GitHub branch containing this directory into a Vercel project, with Root Directory `web`.
2. Connect a **private Vercel Blob store** to the project. This supplies `BLOB_READ_WRITE_TOKEN` as a server environment variable.
3. Add protected server variables `GEMINI_API_KEY` and a random `RATE_LIMIT_SECRET`. Set `GEMINI_MODEL=gemini-3.5-flash-lite`.
4. Deploy to production and check `/api/health` and a new message. The public production alias must open without Vercel Authentication.

Do not put credentials into the public directory, source code, `vercel.json`, or Git. `.env.example` documents names only. `npm ci`, `npm run build`, and `npm test` run the build checks and migration tests.

## Persistent usage limits

A private blob stores quota counters and duplicate-request IDs only. Reads bypass the CDN cache with `useCache: false`; writes use `ifMatch` ETags. Concurrent writes retry on version conflicts, so multiple function instances cannot silently lose increments. Failed storage access prevents unmetered model calls.

Limits are 5 requests/minute and 30/day per hashed IP, and 15/minute and 200/day globally. The quotas count reserved attempts, including provider failures. Expired counters and IDs are pruned during the next quota update. No message text, policy, model reply, or visitor review is persisted in Blob. Google receives the message, bounded context, and retrieved policy passages after limited redaction. Use fictional data in the public demo.

This is a portfolio prototype with operational safeguards, not an integration with a real company's accounts or support system. The confidence score is uncalibrated, valid citations do not guarantee semantic grounding, and no new benchmark accuracy is claimed for the live version.
