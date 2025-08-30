import { createChatBotMessage } from "react-chatbot-kit";

class ActionProvider {
  constructor(createChatBotMessage, setStateFunc) {
    this.createChatBotMessage = createChatBotMessage;
    this.setState = setStateFunc;
  }

  // Handle user message by sending it to backend via GET with query parameter
  async handleUserMessage(message) {
      try {
          const encodedMessage = encodeURIComponent(message);
          const url = `http://localhost:8000/chat?query=${encodedMessage}`;
          console.log("Fetching from:", url);
          
          const response = await fetch(url, {
              method: "GET",
              headers: {
                  "Content-Type": "application/json",
              },
          });

          console.log("Response status:", response.status);
          
          if (!response.ok) {
              throw new Error(`HTTP error! status: ${response.status}`);
          }

          const data = await response.json();
          console.log("Response data:", data);

          const botMessage = this.createChatBotMessage(data.response || data.reply || "No response from server.");
          
          this.setState((prev) => ({
              ...prev,
              messages: [...prev.messages, botMessage],
          }));
      } catch (error) {
          console.error("Detailed error:", error);
          const botMessage = this.createChatBotMessage("⚠️ Server connection failed. Check console for details.");
          this.setState((prev) => ({
              ...prev,
              messages: [...prev.messages, botMessage],
          }));
      }
  }
}

export default ActionProvider;