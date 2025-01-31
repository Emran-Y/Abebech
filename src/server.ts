import express from "express";
import cors from "cors";
import { PrismaClient } from '@prisma/client';
import { processQuestion } from './service/chat-service';
import dotenv from "dotenv";
dotenv.config();

const app = express();
const prisma = new PrismaClient();

// Use CORS middleware to allow all origins
app.use(cors());

// Parse JSON bodies
app.use(express.json());

app.get("/", (req: any, res: any) => {
  res.send("Hello World");
});

app.post('/api/chat/:userId', async (req: any, res: any) => {
  const { userId } = req.params;
  const { content } = req.body;

  if (!userId || !content) {
    return res.status(400).json({ error: 'Missing userId or content' });
  }

  try {
    // Create user message
    const userMessage = await prisma.message.create({
      data: {
        content,
        role: 'user',
        userId,
      },
    });

    // Process with AI
    const aiResponse = await processQuestion(content, userId);

    // Save AI response
    const assistantMessage = await prisma.message.create({
      data: {
        content: aiResponse,
        role: 'assistant',
        userId,
      },
    });

    return res.json(assistantMessage);
  } catch (error) {
    console.error('Error processing chat:', error);
    return res.status(500).json({ error: 'Internal Server Error' });
  }
});

app.get('/api/chat/:userId', async (req: any, res: any) => {
  const { userId } = req.params;

  if (!userId) {
    return res.status(400).json({ error: 'Missing userId' });
  }

  try {
    const messages = await prisma.message.findMany({
      where: {
        userId,
      },
      orderBy: { createdAt: 'asc' },
    });

    return res.json(messages);
  } catch (error) {
    console.error('Error retrieving messages:', error);
    return res.status(500).json({ error: 'Internal Server Error' });
  }
});

// Handle undefined routes
app.use((req: any, res: any) => {
  res.status(404).json({ error: 'Route not found' });
});

const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});