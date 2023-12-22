import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import * as fs from 'fs';
import { Document } from 'langchain/document';
import { BaseDocumentLoader } from 'langchain/document_loaders/base';
import path from 'path';
import { AutodocRepoConfig } from '../../../types.js';
import { HNSWLib } from '../../../langchain/hnswlib.js';

async function processFile(filePath: string): Promise<Document> {
  return await new Promise<Document>((resolve, reject) => {
    fs.readFile(filePath, 'utf8', (err, fileContents) => {
      if (err) {
        reject(err);
      } else {
        const metadata = { source: filePath };
        const doc = new Document({
          pageContent: fileContents,
          metadata: metadata,
        });
        resolve(doc);
      }
    });
  });
}

async function processDirectory(directoryPath: string): Promise<Document[]> {
  const docs: Document[] = [];
  let files: string[];
  try {
    files = fs.readdirSync(directoryPath);
  } catch (err) {
    console.error(err);
    throw new Error(
      `Could not read directory: ${directoryPath}. Did you run \`sh download.sh\`?`,
    );
  }
  for (const file of files) {
    const filePath = path.join(directoryPath, file);
    const stat = fs.statSync(filePath);
    if (stat.isDirectory()) {
      const newDocs = processDirectory(filePath);
      const nestedDocs = await newDocs;
      docs.push(...nestedDocs);
    } else {
      const newDoc = processFile(filePath);
      const doc = await newDoc;
      docs.push(doc);
    }
  }
  return docs;
}

class RepoLoader extends BaseDocumentLoader {
  constructor(public filePath: string) {
    super();
  }
  async load(): Promise<Document[]> {
    return await processDirectory(this.filePath);
  }
}
// : https://test-openai-sandbox.openai.azure.com/openai/deployments/gpt-4-32k/chat/completions?api-version=2023-07-01-preview
export const createVectorStore = async ({
  root,
  output,
}: AutodocRepoConfig): Promise<void> => {
  const loader = new RepoLoader(root);
  const rawDocs = await loader.load();
  /* Split the text into chunks */
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 8000,
    chunkOverlap: 100,
  });
  const docs = await textSplitter.splitDocuments(rawDocs);
  /* Create the vectorstore, https://js.langchain.com/docs/integrations/text_embedding/azure_openai */
  const vectorStore = await HNSWLib.fromDocuments(
    docs,
    new OpenAIEmbeddings({
      azureOpenAIApiKey: process.env.OPENAI_API_KEY,
      azureOpenAIApiVersion: '2023-07-01-preview',
      azureOpenAIApiInstanceName: 'test-openai-sandbox',
      azureOpenAIApiDeploymentName: 'gpt-4-32k',
    }),
  );
  await vectorStore.save(output);
};
