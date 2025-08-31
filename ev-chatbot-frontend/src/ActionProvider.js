import { createChatBotMessage } from "react-chatbot-kit";

class ActionProvider {
  constructor(createChatBotMessage, setStateFunc) {
    this.createChatBotMessage = createChatBotMessage;
    this.setState = setStateFunc;
    // Trigger location request right away
    this.getUserLocation();
  }

  // Method to get user's location
  async getUserLocation() {
    return new Promise((resolve, reject) => {
      if (!navigator.geolocation) {
        reject("Geolocation is not supported by your browser");
      } else {
        navigator.geolocation.getCurrentPosition(
          (position) => {
            resolve({
              lat: position.coords.latitude,
              lon: position.coords.longitude,
            });
          },
          (err) => reject(err.message)
        );
      }
    });
  }
  


  // Handle user message by sending it to backend via GET with query parameter
  async handleUserMessage(message) {
    try {
        // 1️⃣ Add a temporary "Loading..." message
        const loadingMessage = this.createChatBotMessage("⏳ Loading response...");
        this.setState((prev) => ({
            ...prev,
            messages: [...prev.messages, loadingMessage],
        }));

        const encodedMessage = encodeURIComponent(message);
        let url = `http://localhost:8000/chat?query=${encodedMessage}`;
        console.log("Fetching from:", url);


        // If user wants "nearby", add location
        if (message.toLowerCase().includes("nearby") || message.toLowerCase().includes("near me")) {
            try {
                const location = await this.getUserLocation();
                url = `${url}&lat=${location.lat}&lon=${location.lon}`;
            } catch (err) {
                console.warn("Could not get location:", err);
            }
        }

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

        const botMessage = this.createChatBotMessage(
            data.response || data.reply || "No response from server."
        );

        // 2️⃣ Replace the "Loading..." message with the real response
        this.setState((prev) => ({
            ...prev,
            messages: [
            ...prev.messages.slice(0, -1), // remove last message (loading)
            botMessage,
            ],
        }));
    } catch (error) {
        console.error("Detailed error:", error);
        const botMessage = this.createChatBotMessage(
            "⚠️ Server connection failed. Check console for details."
        );
        this.setState((prev) => ({
            ...prev,
            messages: [...prev.messages, botMessage],
        }));
    }
  }
}

export default ActionProvider;
