"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g = Object.create((typeof Iterator === "function" ? Iterator : Object).prototype);
    return g.next = verb(0), g["throw"] = verb(1), g["return"] = verb(2), typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.processQuestion = processQuestion;
exports.cleanupConversation = cleanupConversation;
var openai_1 = require("@langchain/openai");
var prompts_1 = require("@langchain/core/prompts");
var memory_1 = require("langchain/memory");
var chains_1 = require("langchain/chains");
var weaviate_ts_client_1 = __importDefault(require("weaviate-ts-client"));
var weaviate_1 = require("@langchain/weaviate");
// Store conversation chains by conversationId
var conversationChains = new Map();
var TEMPLATE = "Hi! I'm Abebech, your friendly AI buddy at Addis Ababa University! \uD83C\uDF93\n\n**Please format your responses using markdown for better readability.**\n\n**Previous chat:**\n{history}\n\n**Your message:** {input}\n\n**Context from university database:**\n{context}\n\nI'll help you with accurate information about Addis Ababa University. If your question isn't about AAU or I don't have enough context, I'll let you know.\n\nLet me assist you! \uD83D\uDE0A\n\n---\n\n**Example questions you can ask me:**\n- \"What clubs and student organizations can I join at AAU?\" \uD83C\uDFAD\n- \"Help! When are the library opening hours?\" \uD83D\uDCD6\n- \"What's the process for registering for next semester's classes?\" \u270D\uFE0F\n- \"Tell me about the best study spots on campus!\" \uD83D\uDCDD\n- \"What fun events are happening this week?\" \uD83C\uDF89\n\n**P.S.** Don't forget to check out our [website](https://www.aau.edu.et) for more info! \uD83C\uDF10";
// Example questions you can ask me:
// - "What clubs and student organizations can I join at AAU?" 🎭
// - "Help! When are the library opening hours?" 📖
// - "What's the process for registering for next semester's classes?" ✍️
// - "Tell me about the best study spots on campus!" 📝
// - "What fun events are happening this week?" 🎉
var embeddings = new openai_1.OpenAIEmbeddings({
    apiKey: process.env.OPENAI_API_KEY,
    model: 'text-embedding-ada-002', // You can use any supported OpenAI model
});
var weaviateClient = weaviate_ts_client_1.default.client({
    scheme: process.env.WEAVIATE_SCHEME || 'http',
    host: process.env.WEAVIATE_HOST || 'localhost:8080',
    apiKey: new weaviate_ts_client_1.default.ApiKey(process.env.WEAVIATE_API_KEY || ''),
});
var vectorStore = new weaviate_1.WeaviateStore(embeddings, {
    client: weaviateClient, // Type assertion to avoid version mismatch error
    indexName: 'MyVectorDB',
    textKey: 'text',
    metadataKeys: ['source'],
});
function searchWeaviate(query) {
    return __awaiter(this, void 0, void 0, function () {
        var results;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0: return [4 /*yield*/, vectorStore.similaritySearch(query, 3)];
                case 1:
                    results = _a.sent();
                    return [2 /*return*/, results.map(function (doc) { return doc.pageContent; }).join('\n\n')];
            }
        });
    });
}
function getOrCreateConversationChain(conversationId, openAIApiKey) {
    return __awaiter(this, void 0, void 0, function () {
        var chain, memory, llm;
        return __generator(this, function (_a) {
            chain = conversationChains.get(conversationId);
            if (!chain) {
                memory = new memory_1.BufferMemory({
                    returnMessages: true,
                    memoryKey: "history",
                    inputKey: "input",
                    outputKey: "response",
                });
                llm = new openai_1.ChatOpenAI({
                    openAIApiKey: openAIApiKey,
                    temperature: 0.7,
                    modelName: 'gpt-3.5-turbo',
                });
                chain = new chains_1.ConversationChain({
                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    llm: llm,
                    memory: memory,
                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    prompt: prompts_1.PromptTemplate.fromTemplate(TEMPLATE),
                    verbose: process.env.NODE_ENV === 'development'
                });
                conversationChains.set(conversationId, chain);
            }
            return [2 /*return*/, chain];
        });
    });
}
function trimConversationHistory(chain) {
    return __awaiter(this, void 0, void 0, function () {
        var messages, trimmedMessages, _i, trimmedMessages_1, message;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0: return [4 /*yield*/, chain.memory.chatHistory.getMessages()];
                case 1:
                    messages = _a.sent();
                    if (!(messages.length > 10)) return [3 /*break*/, 6];
                    trimmedMessages = messages.slice(-10);
                    return [4 /*yield*/, chain.memory.chatHistory.clear()];
                case 2:
                    _a.sent();
                    _i = 0, trimmedMessages_1 = trimmedMessages;
                    _a.label = 3;
                case 3:
                    if (!(_i < trimmedMessages_1.length)) return [3 /*break*/, 6];
                    message = trimmedMessages_1[_i];
                    return [4 /*yield*/, chain.memory.chatHistory.addMessage(message)];
                case 4:
                    _a.sent();
                    _a.label = 5;
                case 5:
                    _i++;
                    return [3 /*break*/, 3];
                case 6: return [2 /*return*/];
            }
        });
    });
}
function processQuestion(userQuestion, conversationId) {
    return __awaiter(this, void 0, void 0, function () {
        var openAIApiKey, chain, contextWeaviate, answer, _a, _b, _c, _d, _e, playfulResponse, error_1;
        return __generator(this, function (_f) {
            switch (_f.label) {
                case 0:
                    _f.trys.push([0, 6, , 7]);
                    openAIApiKey = process.env.OPENAI_API_KEY;
                    return [4 /*yield*/, getOrCreateConversationChain(conversationId, openAIApiKey)];
                case 1:
                    chain = _f.sent();
                    return [4 /*yield*/, searchWeaviate(userQuestion)];
                case 2:
                    contextWeaviate = _f.sent();
                    console.log("Context: ".concat(contextWeaviate));
                    console.log("=============================");
                    return [4 /*yield*/, chain.call({
                            input: userQuestion,
                            context: contextWeaviate || 'No relevant context found.'
                        })];
                case 3:
                    answer = _f.sent();
                    // Trim conversation history if needed
                    return [4 /*yield*/, trimConversationHistory(chain)];
                case 4:
                    // Trim conversation history if needed
                    _f.sent();
                    // Log the answer and message history
                    console.log("Answer: ".concat(answer.response));
                    _b = (_a = console).log;
                    _c = "Message History: ".concat;
                    _e = (_d = JSON).stringify;
                    return [4 /*yield*/, chain.memory.chatHistory.getMessages()];
                case 5:
                    _b.apply(_a, [_c.apply("Message History: ", [_e.apply(_d, [_f.sent(), null, 2])])]);
                    playfulResponse = "".concat(answer.response, "\n\n**Remember:** Always feel free to explore more on our [website](https://www.aau.edu.et)! \uD83C\uDF1F");
                    return [2 /*return*/, playfulResponse];
                case 6:
                    error_1 = _f.sent();
                    console.error('Error processing question:', error_1);
                    throw error_1;
                case 7: return [2 /*return*/];
            }
        });
    });
}
// Cleanup function with optional reason parameter
function cleanupConversation(conversationId, reason) {
    if (reason === void 0) { reason = 'deleted'; }
    var chain = conversationChains.get(conversationId);
    if (chain === null || chain === void 0 ? void 0 : chain.memory) {
        chain.memory.chatHistory.clear();
    }
    conversationChains.delete(conversationId);
    console.log("Conversation ".concat(conversationId, " cleaned up (").concat(reason, ")"));
}
