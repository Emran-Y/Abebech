import { ChatOpenAI,OpenAIEmbeddings } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import { BufferMemory } from "langchain/memory";
import { ConversationChain } from "langchain/chains";
import weaviate from 'weaviate-ts-client';
import { WeaviateStore } from '@langchain/weaviate';

// Store conversation chains by conversationId
const conversationChains = new Map<string, ConversationChain>();

const TEMPLATE = `Hi! I'm Abebech, your friendly AI buddy at Addis Ababa University! 🎓

**Please format your responses using markdown for better readability.**

**Previous chat:**
{history}

**Your message:** {input}

**Context from university database:**
{context}

I'll help you with accurate information about Addis Ababa University. If your question isn't about AAU or I don't have enough context, I'll let you know.

Let me assist you! 😊

---

**Example questions you can ask me:**
- "What clubs and student organizations can I join at AAU?" 🎭
- "Help! When are the library opening hours?" 📖
- "What's the process for registering for next semester's classes?" ✍️
- "Tell me about the best study spots on campus!" 📝
- "What fun events are happening this week?" 🎉

**P.S.** Don't forget to check out our [website](https://www.aau.edu.et) for more info! 🌐`;



// Example questions you can ask me:
// - "What clubs and student organizations can I join at AAU?" 🎭
// - "Help! When are the library opening hours?" 📖
// - "What's the process for registering for next semester's classes?" ✍️
// - "Tell me about the best study spots on campus!" 📝
// - "What fun events are happening this week?" 🎉

const embeddings = new OpenAIEmbeddings({
    apiKey: process.env.OPENAI_API_KEY,
    model: 'text-embedding-ada-002',  // You can use any supported OpenAI model
  });
  
  const weaviateClient = weaviate.client({
    scheme: process.env.WEAVIATE_SCHEME || 'http',
    host: process.env.WEAVIATE_HOST || 'localhost:8080',
    apiKey: new weaviate.ApiKey(process.env.WEAVIATE_API_KEY || ''),
  });
  const vectorStore = new WeaviateStore(embeddings, {
    client: weaviateClient as any, // Type assertion to avoid version mismatch error
    indexName: 'MyVectorDB',
    textKey: 'text',
    metadataKeys: ['source'],
  });
  
  
  async function searchWeaviate(query: string) {
    const results = await vectorStore.similaritySearch(query, 3);
    return results.map(doc => doc.pageContent).join('\n\n');
  }

async function getOrCreateConversationChain(
  conversationId: string, 
  openAIApiKey: string
) {
  let chain = conversationChains.get(conversationId);
  
  if (!chain) {
    const memory = new BufferMemory({
      returnMessages: true,
      memoryKey: "history",
      inputKey: "input",
      outputKey: "response",
    });

    const llm = new ChatOpenAI({ 
      openAIApiKey,
      temperature: 0.7,
      modelName: 'gpt-3.5-turbo',
    });

    chain = new ConversationChain({
      
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      llm: llm as any,
      memory,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      prompt: PromptTemplate.fromTemplate(TEMPLATE) as any,
      verbose: process.env.NODE_ENV === 'development'
    });
    
    conversationChains.set(conversationId, chain);
  }

  return chain;
}

async function trimConversationHistory(chain: ConversationChain) {
  const messages = await (chain.memory as BufferMemory).chatHistory.getMessages();
  if (messages.length > 10) {
    // Keep only the last 10 messages
    const trimmedMessages = messages.slice(-10);
    await (chain.memory as BufferMemory).chatHistory.clear();
    for (const message of trimmedMessages) {
      await (chain.memory as BufferMemory).chatHistory.addMessage(message);
    }
  }
}

export async function processQuestion(
  userQuestion: string, 
  conversationId: string
) {
  try {
    const openAIApiKey = process.env.OPENAI_API_KEY!;
    
    // Get or create conversation chain
    const chain = await getOrCreateConversationChain(conversationId, openAIApiKey);
    
    // Get relevant context from vector store
    const contextWeaviate = await searchWeaviate(userQuestion);

    console.log(`Context: ${contextWeaviate}`);
    console.log(`=============================`);

    // Process the question
    const answer = await chain.call({ 
      input: userQuestion,
      context: contextWeaviate || 'No relevant context found.'
    });

    // Trim conversation history if needed
    await trimConversationHistory(chain);

    // Log the answer and message history
    console.log(`Answer: ${answer.response}`);
    console.log(`Message History: ${JSON.stringify(await (chain.memory as BufferMemory).chatHistory.getMessages(), null, 2)}`);

    // Add playful and friendly tone
    const playfulResponse = `${answer.response}\n\n**Remember:** Always feel free to explore more on our [website](https://www.aau.edu.et)! 🌟`;

    return playfulResponse;
  } catch (error) {
    console.error('Error processing question:', error);
    throw error;
  }
}

// Cleanup function with optional reason parameter
export function cleanupConversation(
  conversationId: string, 
  reason: 'deleted' | 'expired' = 'deleted'
) {
  const chain = conversationChains.get(conversationId);
  if (chain?.memory) {
    (chain.memory as BufferMemory).chatHistory.clear();
  }
  conversationChains.delete(conversationId);
  console.log(`Conversation ${conversationId} cleaned up (${reason})`);
}
