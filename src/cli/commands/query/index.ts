import chalk from 'chalk';
import inquirer from 'inquirer';
import { marked, Renderer } from 'marked';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import path from 'path';
import { HNSWLib } from '../../../langchain/hnswlib.js';
import { AutodocRepoConfig, AutodocUserConfig } from '../../../types.js';
import { makeChain } from './createChatChain.js';
import { stopSpinner, updateSpinnerText } from '../../spinner.js';
import TerminalRenderer from 'marked-terminal';
const renderer = new TerminalRenderer({});
const chatHistory: [string, string][] = [];

marked.setOptions({ renderer: renderer as unknown as Renderer<never> });

const displayWelcomeMessage = (projectName: string) => {
  console.log(chalk.bold.blue(`Welcome to the ${projectName} chatbot.`));
  console.log(
    `Ask any questions related to the ${projectName} codebase, and I'll try to help. Type 'exit' to quit.\n`,
  );
};

const clearScreenAndMoveCursorToTop = () => {
  process.stdout.write('\x1B[2J\x1B[0f');
};

export const query = async (
  {
    name,
    repositoryUrl,
    output,
    contentType,
    chatPrompt,
    targetAudience,
  }: AutodocRepoConfig,
  { llms }: AutodocUserConfig,
) => {
  const data = path.join(output, 'docs', 'data/');
  const vectorStore = await HNSWLib.load(
    data,
    new OpenAIEmbeddings({
      azureOpenAIApiKey: process.env.OPENAI_API_KEY,
      azureOpenAIApiVersion: '2023-07-01-preview',
      azureOpenAIApiInstanceName: 'test-openai-sandbox',
      azureOpenAIApiDeploymentName: 'gpt-4-32k',
    }),
  );
  const chain = makeChain(
    name,
    repositoryUrl,
    contentType,
    chatPrompt,
    targetAudience,
    vectorStore,
    llms,
    (token: string) => {
      stopSpinner();
      process.stdout.write(token);
    },
  );

  clearScreenAndMoveCursorToTop();
  displayWelcomeMessage(name);

  const getQuestion = async () => {
    const { question } = await inquirer.prompt([
      {
        type: 'input',
        name: 'question',
        message: chalk.yellow(`How can I help with ${name}?\n`),
      },
    ]);

    return question;
  };

  let question = await getQuestion();

  while (question !== 'exit') {
    updateSpinnerText('Thinking...');
    try {
      const { text } = await chain.call({
        question,
        chat_history: chatHistory,
      });

      chatHistory.push([question, text]);

      console.log('\n\nMarkdown:\n');
      console.log(marked(text));

      question = await getQuestion();
    } catch (error: any) {
      console.log(chalk.red(`Something went wrong: ${error.message}`));
      question = await getQuestion();
    }
  }
};
