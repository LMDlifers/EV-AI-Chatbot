import axios from "axios";

class ActionProvider {
  constructor(createChatBotMessage, setStateFunc) {
    this.createChatBotMessage = createChatBotMessage;
    this.setState = setStateFunc;
  }

  async handleUserMessage(message) {
    try {
      const res = await axios.post("http://localhost:5000/api/chat", { message });
      const botReply = this.createChatBotMessage(res.data.reply);

      this.setState(prev => ({
        ...prev,
        messages: [...prev.messages, botReply],
      }));
    } catch (err) {
      console.error(err);
    }
  }
}

export default ActionProvider;
