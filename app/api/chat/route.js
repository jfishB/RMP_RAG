import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import OpenAI from "openai";

const systemPrompt = `
You are a helpful and knowledgeable agent designed to assist students in finding the best professors based on their specific queries. Using Retrieval-Augmented Generation (RAG), your role is to understand the student's question and provide detailed information on the top 3 professors that best match their needs.

When responding:

Consider the student's preferences and the context of their query, such as course difficulty, teaching style, and overall rating.
Provide a brief summary for each professor, including key strengths, common student feedback, and any relevant course information.
Always prioritize accuracy, clarity, and usefulness of the information.
Example User Query: "Which professors are best for an easy A in Calculus?"

Example Response:

Professor Jane Smith - Highly rated for her clear explanations and generous grading. Students often mention that she provides ample resources for practice and her exams are straightforward.
Professor John Doe - Known for being approachable and providing extra help outside of class. His classes are well-organized, and many students find his grading to be fair and lenient.
Professor Emily Davis - Frequently praised for her engaging lectures and understanding nature. Her exams are manageable if you attend classes regularly and participate in discussions.
Make sure to tailor each response to the student's needs and provide a balanced view of each professor's strengths and areas for improvement.
`;

export async function POST(req) {
  const data = await req.json();
  const pc = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY,
  });
  const index = pc.index("rag").namespace("ns1");
  const openai = new OpenAI();

  const text = data[data.length - 1].content;
  const embedding = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: text,
    encoding_format: "float",
  });

  const results = await index.query({
    topK: 3,
    includeMetadata: true,
    vector: embedding.data[0].embedding,
  });

  let resultString =
    "\n\nReturned results from vector db (done automatically): ";
  results.matches.forEach((match) => {
    resultString += `
        Professor: ${match.id}
        Review: ${match.metadata.stars}
        Subject: ${match.metadata.subject}
        Stars: ${match.metadata.stars}
        \n\n`
  });

  const lastMessage = data[data.length - 1];
  const lastMessageContent = lastMessage.content + resultString;
  const lastDataWithoutLastMessage = data.slice(0, data.length - 1);
  const completion = await openai.chat.completions.create({
    messages: [
      { role: "system", content: systemPrompt },
      ...lastDataWithoutLastMessage,
      { role: "user", content: lastMessageContent },
    ],
    model: "gpt-4o-mini",
    stream: true,
  });

  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder();
      try {
        for await (const chunk of completion) {
          const content = chunk.choices[0]?.delta?.content;
          if (content) {
            const text = encoder.encode(content);
            controller.enqueue(text);
          }
        }
      } catch (err) {
        controller.error(err);
      } finally {
        controller.close();
      }
    },
  });

  return new NextResponse(stream)
}
