import { createChatBotMessage } from "react-chatbot-kit";

const config = {
  botName: "MyBot",
  initialMessages: [createChatBotMessage("Hello! How can I help you?")],
};

export default config;
