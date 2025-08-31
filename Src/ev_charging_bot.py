# imports
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import hdbscan
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import kagglehub
from kagglehub import KaggleDatasetAdapter
from typing import List, Dict, Optional

class EVChargingStationBot:
    """
    AI Chatbot for Electric Vehicle Charging Station queries.
    Supports semantic search, geographic clustering, structured filtering, and natural language responses.
    """
    
    def __init__(self, dataset_handle: str = "sahirmaharajj/electric-vehicle-charging-stations-2024", 
                file_path: str = None, embedding_model: str = 'all-MiniLM-L6-v2',
                decoder_model: str = 'gpt2', use_decoder: bool = True):
        """
        Initialize the EV Charging Station Bot.
        
        Args:
            dataset_handle: Kaggle dataset identifier
            file_path: Specific file to load from dataset
            embedding_model: SentenceTransformer model for semantic search
            decoder_model: Decoder model for natural language generation
            use_decoder: Whether to initialize decoder model (set False for faster startup)
        """
        self.dataset_handle = dataset_handle
        self.file_path = file_path
        self.embedding_model_name = embedding_model
        self.decoder_model_name = decoder_model
        self.use_decoder = use_decoder
        
        # Core data structures
        self.df = None
        self.model = None
        self.station_embeddings = None
        self.geo_clusters = None
        
        # Decoder components
        self.tokenizer = None
        self.decoder_model = None
        
        # Initialize the bot
        self._load_data()
        self._preprocess_data()
        self._initialize_embeddings()
        if self.use_decoder:
            self._initialize_decoder()
        
    def _load_data(self) -> None:
        """Load the EV charging station dataset."""
        try:
            if self.file_path:
                self.df = kagglehub.dataset_load(
                    KaggleDatasetAdapter.PANDAS,
                    self.dataset_handle,
                    self.file_path
                )
            else:
                # Download and explore dataset first
                path = kagglehub.dataset_download(self.dataset_handle)
                print(f"Dataset downloaded to: {path}")
                # User needs to specify the correct file_path
                raise ValueError("Please specify the file_path after exploring the dataset")
                
            print(f"Loaded {len(self.df)} charging stations")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def _preprocess_data(self) -> None:
        """Preprocess the dataset for optimal searching."""
        # Clean and standardize columns FIRST
        self._clean_data()
        
        # Extract coordinates from georeferenced column
        if 'New Georeferenced Column' in self.df.columns:
            coordinates = self.df['New Georeferenced Column'].str.extract(r'POINT \(([^)]+)\)')
            coord_split = coordinates[0].str.split(' ', expand=True)
            if len(coord_split.columns) >= 2:
                self.df['longitude'] = pd.to_numeric(coord_split[0], errors='coerce')
                self.df['latitude'] = pd.to_numeric(coord_split[1], errors='coerce')
        
        # Create rich descriptions for semantic search (after cleaning)
        self.df['description'] = self.df.apply(self._create_station_description, axis=1)
        
        print("Data preprocessing completed")
    
    def _create_station_description(self, row) -> str:
        """Create a rich text description for each charging station."""
        description_parts = []
        
        if pd.notna(row.get('Station Name')):
            description_parts.append(f"Station: {row['Station Name']}")
        
        if pd.notna(row.get('City')) and pd.notna(row.get('Street Address')):
            description_parts.append(f"Located in {row['City']} at {row['Street Address']}")
        
        if pd.notna(row.get('Access Days Time')):
            description_parts.append(f"Hours: {row['Access Days Time']}")
        
        # Charging capabilities
        charging_info = []
        if pd.notna(row.get('EV Level1 EVSE Num')) and row['EV Level1 EVSE Num'] > 0:
            charging_info.append(f"Level 1: {row['EV Level1 EVSE Num']} ports")
        if pd.notna(row.get('EV Level2 EVSE Num')) and row['EV Level2 EVSE Num'] > 0:
            charging_info.append(f"Level 2: {row['EV Level2 EVSE Num']} ports")
        if pd.notna(row.get('EV DC Fast Count')) and row['EV DC Fast Count'] > 0:
            charging_info.append(f"DC Fast: {row['EV DC Fast Count']} ports")
        
        if charging_info:
            description_parts.append("Charging: " + ", ".join(charging_info))
        
        return ". ".join(description_parts)
    
    def _clean_data(self) -> None:
        """Clean and standardize the dataset."""
        # Convert numeric columns - handle "NONE" strings
        numeric_cols = ['EV Level1 EVSE Num', 'EV Level2 EVSE Num', 'EV DC Fast Count']
        for col in numeric_cols:
            if col in self.df.columns:
                # Replace "NONE" with 0, then convert to numeric
                self.df[col] = self.df[col].astype(str).str.replace('NONE', '0')
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0).astype(int)
        
        # Clean string columns
        string_cols = ['Station Name', 'City', 'Street Address']
        for col in string_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.strip()
    
    def _initialize_embeddings(self) -> None:
        """Initialize the sentence transformer model and create embeddings."""
        try:
            self.model = SentenceTransformer(self.embedding_model_name)
            self.station_embeddings = self.model.encode(self.df['description'].tolist())
            print("Embeddings initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize embeddings: {e}")
            self.model = None
            self.station_embeddings = None
    
    def _initialize_decoder(self) -> None:
        """Initialize the decoder model for natural language generation."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.decoder_model_name)
            self.decoder_model = AutoModelForCausalLM.from_pretrained(self.decoder_model_name)
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("Decoder model initialized for response generation")
        except Exception as e:
            print(f"Warning: Could not initialize decoder: {e}")
            self.tokenizer = None
            self.decoder_model = None
    
    # Hyperparameter tuning not in used due for performance's sake. But can use Hyperopt for significant improvement compared to GridSearch
    def setup_geographic_clustering(self, min_cluster_size: int = 5, min_samples: int = 3) -> None:
        """Setup geographic clustering for location-based optimization using HDBSCAN."""
        if 'latitude' in self.df.columns and 'longitude' in self.df.columns:
            # Remove rows with missing coordinates
            valid_coords = self.df.dropna(subset=['latitude', 'longitude'])
            
            if len(valid_coords) > 0:
                from sklearn.preprocessing import StandardScaler
                import hdbscan
                
                coords = valid_coords[['latitude', 'longitude']].values
                
                # Standardize coordinates
                scaler = StandardScaler()
                coords_scaled = scaler.fit_transform(coords)
                
                # Apply HDBSCAN clustering
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    metric='euclidean'
                )
                cluster_labels = clusterer.fit_predict(coords_scaled)
                
                # Map clusters back to original dataframe
                self.df['geo_cluster'] = -1  # Default for missing coordinates
                self.df.loc[valid_coords.index, 'geo_cluster'] = cluster_labels
                
                self.geo_clusters = clusterer
                
                # Print results
                n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                n_noise = list(cluster_labels).count(-1)
                print(f"HDBSCAN clustering completed: {n_clusters} clusters, {n_noise} outliers")
            else:
                print("No valid coordinates found for clustering")
        else:
            print("Latitude/Longitude columns not available for clustering")
    
    # === BASIC SEARCH METHODS ===
    def find_by_city(self, city: str, limit: int = 10) -> pd.DataFrame:
        """Find charging stations by city name."""
        mask = self.df['City'].str.contains(city, case=False, na=False)
        return self.df[mask].head(limit)
    
    def find_fast_charging(self, city: str = None, limit: int = 10) -> pd.DataFrame:
        """Find stations with DC fast charging."""
        # Convert to numeric on the fly if needed
        dc_fast_col = pd.to_numeric(self.df['EV DC Fast Count'], errors='coerce').fillna(0)
        mask = dc_fast_col > 0
        
        if city:
            city_mask = self.df['City'].str.contains(city, case=False, na=False)
            mask = mask & city_mask
        
        return self.df[mask].head(limit)

    def find_level2_charging(self, city: str = None, min_ports: int = 1, limit: int = 10) -> pd.DataFrame:
        """Find stations with Level 2 charging."""
        # Convert to numeric on the fly if needed
        level2_col = pd.to_numeric(self.df['EV Level2 EVSE Num'], errors='coerce').fillna(0)
        mask = level2_col >= min_ports
        
        if city:
            city_mask = self.df['City'].str.contains(city, case=False, na=False)
            mask = mask & city_mask
        
        return self.df[mask].head(limit)
    
    def find_24_hour_stations(self, city: str = None, limit: int = 10) -> pd.DataFrame:
        """Find 24-hour accessible charging stations."""
        mask = self.df['Access Days Time'].str.contains('24 hours', case=False, na=False)
        
        if city:
            city_mask = self.df['City'].str.contains(city, case=False, na=False)
            mask = mask & city_mask
        
        return self.df[mask].head(limit)
    
    def find_nearby_stations(self, lat: float, lon: float, radius_km: float = 10, limit: int = 10) -> pd.DataFrame:
        """Find charging stations within a radius of given coordinates."""
        if 'latitude' not in self.df.columns or 'longitude' not in self.df.columns:
            raise ValueError("Coordinate data not available")
        
        # Calculate distances using Haversine formula
        distances = self._calculate_distances(lat, lon)
        nearby_mask = distances <= radius_km
        
        # Sort by distance
        nearby_stations = self.df[nearby_mask].copy()
        nearby_stations['distance_km'] = distances[nearby_mask]
        
        return nearby_stations.sort_values('distance_km').head(limit)
    
    def semantic_search(self, query: str, limit: int = 5) -> pd.DataFrame:
        """Perform semantic search using embeddings."""
        if self.model is None or self.station_embeddings is None:
            raise ValueError("Embeddings not available. Initialize embeddings first.")
        
        # Encode the query
        query_embedding = self.model.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.station_embeddings)[0]
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:limit]
        results = self.df.iloc[top_indices].copy()
        results['similarity_score'] = similarities[top_indices]
        
        return results
    
    def _prepare_context(self, results_df: pd.DataFrame, search_type: str) -> str:
        """
        Prepare structured context from search results.
        """
        if results_df.empty:
            return "No stations found matching your criteria."
        
        context_parts = []
        context_parts.append(f"Found {len(results_df)} charging stations:")
        
        for idx, row in results_df.iterrows():
            station_info = []
            station_info.append(f"- {row['Station Name']}")
            station_info.append(f"Address: {row['Street Address']}, {row['City']}")
            station_info.append(f"Hours: {row['Access Days Time']}")
            
            # Add charging details
            charging_details = []
            if row['EV Level2 EVSE Num'] > 0:
                charging_details.append(f"Level 2: {row['EV Level2 EVSE Num']} ports")
            if row['EV DC Fast Count'] > 0:
                charging_details.append(f"DC Fast: {row['EV DC Fast Count']} ports")
            
            if charging_details:
                station_info.append(f"Charging: {', '.join(charging_details)}")
            
            # Add distance if available
            if 'distance_km' in row:
                station_info.append(f"Distance: {row['distance_km']:.1f} km")
            
            context_parts.append(" | ".join(station_info))
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str, search_type: str, conversation_context: Dict = None) -> str:
        """Create an intelligent prompt that lets LLM handle conversation flow."""
        
        # System context about the assistant's capabilities
        system_context = f"""You are an intelligent EV charging station assistant. You can help users find charging stations using these actions:

    AVAILABLE ACTIONS:
    - SEARCH_DC_FAST: Find DC fast charging stations (20-60 min charge)
    - SEARCH_LEVEL2: Find Level 2 stations (2-8 hour charge) 
    - SEARCH_LEVEL1: Find Level 1 stations (8-12 hour charge)
    - SEARCH_NEARBY: Find stations near coordinates
    - ASK_CHARGER_TYPE: Ask user to specify charging preference
    - PROVIDE_DIRECTIONS: Give navigation to specific station

    CURRENT LOCATION: {conversation_context.get('location', {}).get('name', 'New York City')} 
    USER CONTEXT: {conversation_context if conversation_context else 'New conversation'}

    CONVERSATION HISTORY: {context}

    USER QUERY: "{query}"

    INSTRUCTIONS:
    1. Understand what the user needs
    2. If charging type is unclear, ask for preference with options
    3. If user specifies or implies urgency, suggest DC Fast charging
    4. When presenting station options, number them 1-5 for easy selection
    5. For station selection (numbers 1-5), provide navigation and tips
    6. Be conversational and helpful
    7. Always mention distance and charging details

    RESPOND WITH:"""

        return system_context
    
    def _decode_response(self, prompt: str, max_length: int = 300, temperature: float = 0.7) -> str:
        """Generate response with conversation awareness."""
        
        # Add conversation instructions to prompt
        enhanced_prompt = f"""{prompt}

    RESPONSE GUIDELINES:
    - Be helpful and conversational
    - If user needs charging, ask about urgency/type if unclear
    - Provide numbered options when showing stations
    - Include practical details (distance, ports, hours)
    - Use emojis appropriately for better UX
    - End with clear next steps for the user

    Response:"""
        
        # [Keep existing generation code but use enhanced_prompt]
        inputs = self.tokenizer.encode(enhanced_prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Generate response using different decoding strategies
        with torch.no_grad():
            outputs = self.decoder_model.generate(
                inputs,
                max_length=inputs.shape[1] + max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Decode and clean the response
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (remove the prompt)
        response = generated_text[len(prompt):].strip()
        
        return response if response else "I found the stations listed above. Please let me know if you need more specific directions!"
    
    def _generate_intelligent_response(self, query: str, context: str, conversation_context: Dict) -> Dict:
        """Let LLM generate response and determine actions."""
        
        # Enhanced prompt for conversation management
        prompt = self._create_prompt(query, context, "intelligent", conversation_context)
        
        # Get LLM response
        llm_response = self._decode_response(prompt, max_length=300)
        
        # Parse LLM response for actions
        actions = self._parse_llm_actions(llm_response, query, conversation_context)
        
        return {
            'response': llm_response,
            'context': conversation_context,
            'actions': actions,
            'data': actions.get('search_results', [])
        }

    def _parse_llm_actions(self, llm_response: str, user_query: str, context: Dict) -> Dict:
        """Parse LLM response to extract actionable commands."""
        actions = {'type': 'response_only'}
        
        # Check if LLM suggests searching (you can train it to use keywords)
        response_lower = llm_response.lower()
        
        if 'search_dc_fast' in response_lower or ('fast' in response_lower and 'charging' in response_lower):
            actions = self._execute_search('dc_fast', context)
        elif 'search_level2' in response_lower or ('level 2' in response_lower):
            actions = self._execute_search('level2', context)
        elif 'search_level1' in response_lower or ('level 1' in response_lower):
            actions = self._execute_search('level1', context)
        elif any(char.isdigit() for char in user_query) and 'station' in llm_response.lower():
            # User selected a station number
            actions = self._handle_station_selection(user_query, context)
        
        return actions
        
    def _simple_template_decoder(self, query: str, results_df: pd.DataFrame, search_type: str) -> str:
        """Enhanced template-based response generation."""
        query_lower = query.lower()
        
        # Handle greetings and general queries FIRST
        if any(word in query_lower for word in ['hi', 'hello', 'hey', 'help']) or search_type in ['greeting', 'template']:
            return """👋 Hello! I'm your EV charging assistant. I can help you find:

    🔌 **Charging Options:**
    • DC Fast Charging (20-60 min) - Perfect for road trips
    • Level 2 Charging (2-8 hours) - Great for shopping/work
    • Level 1 Charging (8-12 hours) - Good for overnight

    📍 **Location Services:**
    • Find stations near you
    • Search by city
    • 24-hour accessible stations

    Just tell me what you need! For example:
    • "I need fast charging"
    • "Find stations in Seattle" 
    • "I need to charge my car urgently"

    What can I help you find today?"""

        # Handle charging requests
        if any(word in query_lower for word in ['charge', 'charging', 'battery', 'power', 'station']):
            if any(word in query_lower for word in ['fast', 'quick', 'urgent', 'emergency', 'dc']):
                return """⚡ **Fast charging it is!** 

    I'll find DC fast charging stations near New York City for you. These can charge your car in 20-60 minutes.

    Let me search for the best options... 

    Would you like me to:
    1. Show you the closest fast charging stations
    2. Find stations with the most charging ports
    3. Look for 24-hour accessible stations

    Just let me know your preference!"""
            
            elif any(word in query_lower for word in ['level 2', 'l2', 'standard', 'regular']):
                return """🔋 **Level 2 charging is a great choice!**

    I'll find Level 2 charging stations near New York City. These typically take 2-8 hours for a full charge - perfect for shopping, work, or dining.

    Searching for the best Level 2 options for you...

    Would you prefer:
    1. Stations with the most Level 2 ports
    2. 24-hour accessible locations
    3. Stations near shopping centers

    Let me know what works best for you!"""
            
            elif any(word in query_lower for word in ['level 1', 'l1', 'slow', 'overnight']):
                return """🏠 **Level 1 charging - perfect for overnight!**

    I'll find Level 1 charging stations near New York City. These are slower (8-12 hours) but often free and great for overnight charging.

    Searching for Level 1 options...

    These are typically found at:
    • Hotels and motels
    • Some workplaces
    • Residential areas
    • Free public locations

    Would you like me to show you the available options?"""
            
            else:
                return """🔌 **I can help you find charging stations!**

    What type of charging do you need?

    1. **DC Fast Charging** ⚡ (20-60 mins) - Great when you're in a hurry
    2. **Level 2 Charging** 🔋 (2-8 hours) - Perfect for longer stops
    3. **Level 1 Charging** 🏠 (8-12 hours) - Good for overnight

    Just reply with the number or tell me about your situation!"""
        
        # Handle number selections (1, 2, 3)
        if query.strip() in ['1', '2', '3']:
            if query.strip() == '1':
                return """⚡ **Searching for DC Fast Charging stations...**

    Finding the closest and fastest charging options near New York City. These will get you back on the road quickly!

    Please wait while I locate the best DC fast charging stations for you..."""
            elif query.strip() == '2':
                return """🔋 **Searching for Level 2 Charging stations...**

    Finding convenient Level 2 charging options near New York City. Perfect for while you shop, work, or dine!

    Please wait while I find the best Level 2 stations for you..."""
            elif query.strip() == '3':
                return """🏠 **Searching for Level 1 Charging stations...**

    Finding overnight and slow charging options near New York City. Great for extended stays!

    Please wait while I locate Level 1 charging stations for you..."""
        
        # Handle station results
        if not results_df.empty:
            station_count = len(results_df)
            first_station = results_df.iloc[0]
            
            # Get charging info for first station
            dc_fast_count = first_station.get('EV DC Fast Count', 0)
            level2_count = first_station.get('EV Level2 EVSE Num', 0)
            level1_count = first_station.get('EV Level1 EVSE Num', 0)
            
            charging_info = []
            if dc_fast_count > 0:
                charging_info.append(f"{dc_fast_count} DC Fast")
            if level2_count > 0:
                charging_info.append(f"{level2_count} Level 2")
            if level1_count > 0:
                charging_info.append(f"{level1_count} Level 1")
            
            charging_text = " + ".join(charging_info) if charging_info else "Multiple types"
            
            if search_type == "semantic":
                return f"""✅ I found {station_count} stations matching your request!

    **🏆 Top Result:** {first_station['Station Name']}
    📍 **Address:** {first_station['Street Address']}, {first_station['City']}
    🕒 **Hours:** {first_station['Access Days Time']}
    🔌 **Charging:** {charging_text} ports
    {'🚗 **Distance:** ' + str(round(first_station['distance_km'], 1)) + ' km away' if 'distance_km' in first_station else ''}

    **What would you like to do next?**
    • Get directions to this station
    • See more station options
    • Get details about charging speeds
    • Find different type of charging

    How can I help you get charged up? 🚗⚡"""

            elif search_type == "location":
                return f"""📍 Found {station_count} charging stations in your area!

    **🎯 Closest Station:** {first_station['Station Name']}
    📍 **Address:** {first_station['Street Address']}, {first_station['City']}
    🚗 **Distance:** {first_station.get('distance_km', 0):.1f} km away
    🕒 **Hours:** {first_station['Access Days Time']}
    🔌 **Available:** {charging_text} ports

    **Ready to go?**
    • Get turn-by-turn directions
    • See all {station_count} nearby options
    • Filter by charging type
    • Check real-time availability

    Shall I provide directions to get you there? 🗺️"""

            elif search_type == "fast_charging":
                return f"""⚡ Excellent! Found {station_count} fast charging stations!

    **🚀 Featured Station:** {first_station['Station Name']}
    📍 **Location:** {first_station['Street Address']}, {first_station['City']}
    ⚡ **Fast Charging:** {dc_fast_count} DC Fast ports available
    🕒 **Open:** {first_station['Access Days Time']}
    {'🚗 **Distance:** ' + str(round(first_station['distance_km'], 1)) + ' km' if 'distance_km' in first_station else ''}

    **⚡ Quick Charge Benefits:**
    • 10-80% charge in 20-45 minutes
    • Perfect for road trips
    • Get back on the road fast!

    **Next steps:**
    • Navigate to this station
    • See all fast charging options
    • Check charging network (Tesla, Electrify America, etc.)

    Ready to charge up quickly? 🔌"""

            elif search_type == "level2_charging":
                return f"""🔋 Perfect! Found {station_count} Level 2 charging stations!

    **🎯 Recommended Station:** {first_station['Station Name']}
    📍 **Location:** {first_station['Street Address']}, {first_station['City']}
    🔋 **Level 2 Ports:** {level2_count} available
    🕒 **Hours:** {first_station['Access Days Time']}
    {'🚗 **Distance:** ' + str(round(first_station['distance_km'], 1)) + ' km' if 'distance_km' in first_station else ''}

    **🔋 Level 2 Benefits:**
    • 2-8 hour full charge
    • Often cheaper than fast charging
    • Great for longer stays

    **Perfect for:**
    • Shopping trips • Work days • Dining out • Movies

    **What's next?**
    • Get directions to this station
    • See all Level 2 options nearby
    • Check if payment required

    Ready to plug in? 🔌"""

            else:
                # Generic results display
                return f"""✅ Great! I found {station_count} charging stations for you!

    **🏆 Top Match:** {first_station['Station Name']}
    📍 {first_station['Street Address']}, {first_station['City']}
    🔌 {charging_text} ports
    🕒 {first_station['Access Days Time']}

    Would you like directions to this station or see more options? 🚗⚡"""
        
        # Handle specific search terms
        if any(word in query_lower for word in ['near', 'nearby', 'close', 'distance']):
            return """📍 **I can find nearby charging stations!**

    To give you the most accurate results, I'm using your current location as New York City.

    What type of nearby stations are you looking for?
    • ⚡ Fast charging (DC Fast)
    • 🔋 Standard charging (Level 2)  
    • 🏠 Slow charging (Level 1)
    • 🌙 24-hour accessible stations

    Just let me know your preference and I'll show you the closest options!"""
        
        if any(word in query_lower for word in ['24', 'hour', 'late', 'night', 'always', 'open']):
            return """🌙 **Looking for 24-hour charging stations!**

    Great choice for peace of mind! I'll find stations that are accessible anytime, day or night.

    These are perfect for:
    • Late night arrivals
    • Early morning departures  
    • Emergency charging situations
    • Flexible travel schedules

    Let me search for 24-hour accessible stations near New York City..."""
        
        if any(word in query_lower for word in ['tesla', 'supercharger']):
            return """🚗 **Tesla Supercharger stations!**

    Looking for Tesla-specific charging! Tesla Superchargers are some of the fastest and most reliable options.

    **Tesla Charging Options:**
    • ⚡ Supercharger V3 (250kW) - Fastest option
    • ⚡ Supercharger V2 (150kW) - Still very fast  
    • 🔌 Destination Chargers - At hotels/restaurants

    I'll search for Tesla-compatible fast charging stations near New York City..."""
        
        if any(word in query_lower for word in ['free', 'cost', 'price', 'cheap']):
            return """💰 **Looking for affordable charging options!**

    Smart thinking! Let me help you find cost-effective charging solutions.

    **Money-saving tips:**
    • Level 1 charging often free at hotels/businesses
    • Some Level 2 stations offer free charging
    • Check apps like PlugShare for free locations
    • Workplace charging often discounted

    I'll prioritize stations with lower costs or free charging options..."""
        
        # Handle city searches
        if any(word in query_lower for word in ['city', 'town', 'in ']):
            return """🏙️ **City-specific search!**

    I can help you find charging stations in specific cities! 

    Just tell me:
    • Which city you're interested in
    • What type of charging you need
    • Any specific requirements (24-hour, free, fast, etc.)

    For example: "Find fast charging in Seattle" or "Level 2 stations in Portland"

    What city would you like to explore for charging options?"""
        
        # Handle fallback cases
        if search_type == 'fallback':
            return """🔌 **I'm here to help with EV charging!**

    I can assist you with:
    • Finding charging stations by location
    • Searching by charging type (Fast, Level 2, Level 1)
    • Locating 24-hour accessible stations
    • Getting directions and station details

    Try asking me something like:
    • "I need fast charging"
    • "Find stations in [city name]"
    • "Show me nearby charging options"
    • "I need to charge my car urgently"

    What can I help you find today?"""
        
        # Default fallback for unrecognized queries
        return """🤖 **I'm your EV charging assistant!**

    I didn't quite understand that request, but I'm here to help you find charging stations!

    **Popular requests:**
    • "I need fast charging" - For quick stops
    • "Find Level 2 charging" - For longer stays
    • "Show nearby stations" - Based on your location
    • "24-hour charging" - Always accessible options

    **You can also ask about:**
    • Specific cities or locations
    • Tesla Superchargers
    • Free charging options
    • Station details and directions

    What would you like to know about EV charging? 🚗⚡"""
    
    def _fallback_template_response(self, user_query: str, conversation_context: Dict) -> Dict:
        """Fallback response when decoder is not available."""
        # Handle greetings FIRST, before trying semantic search
        query_lower = user_query.lower()
        
        if any(word in query_lower for word in ['hi', 'hello', 'hey', 'help']):
            response = self._simple_template_decoder(user_query, pd.DataFrame(), 'greeting')
        else:
            # Try semantic search for other queries
            try:
                results = self.semantic_search(user_query, limit=3)
                response = self._simple_template_decoder(user_query, results, 'semantic')
            except:
                response = self._simple_template_decoder(user_query, pd.DataFrame(), 'fallback')
        
        return {
            'response': response,
            'context': conversation_context,
            'actions': {'type': 'template_response'}
        }
    
    # === UTILITY METHODS ===
    
    def _calculate_distances(self, lat: float, lon: float) -> np.ndarray:
        """Calculate distances using Haversine formula."""
        R = 6371  # Earth's radius in kilometers
        
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        station_lats = np.radians(self.df['latitude'].fillna(0))
        station_lons = np.radians(self.df['longitude'].fillna(0))
        
        dlat = station_lats - lat_rad
        dlon = station_lons - lon_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat_rad) * np.cos(station_lats) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def get_station_details(self, station_id: int) -> Dict:
        """Get detailed information about a specific station."""
        station = self.df.iloc[station_id]
        return station.to_dict()
    
    def get_cities(self) -> List[str]:
        """Get list of all cities with charging stations."""
        return sorted(self.df['City'].unique())
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics about the dataset."""
        return {
            'total_stations': len(self.df),
            'cities': len(self.df['City'].unique()),
            'level1_stations': (self.df['EV Level1 EVSE Num'] > 0).sum(),
            'level2_stations': (self.df['EV Level2 EVSE Num'] > 0).sum(),
            'fast_dc_stations': (self.df['EV DC Fast Count'] > 0).sum(),
            'total_level2_ports': self.df['EV Level2 EVSE Num'].sum(),
            'total_fast_dc_ports': self.df['EV DC Fast Count'].sum()
        }

    def chat_assistant(self, user_query: str, lat = None, lon = None) -> Dict:
        """Simple, working chat assistant that actually executes searches."""
        if lat is not None and lon is not None:
            conversation_context = {
                'location': {'lat': lat, 'lon': lon },
            }
        else:
            conversation_context = {
                'location': {'lat': 40.7128, 'lon': -74.0060 },
            }
        
        query_lower = user_query.lower().strip()
        
        # === SEARCH TRIGGERS ===
        
        # Numbers 1, 2, 3 - FORCE SEARCH
        if user_query.strip() in ['1', '2', '3']:
            print(f"🚀 Number {user_query.strip()} selected - executing search")
            
            if user_query.strip() == '1':
                return self._force_search_with_results("dc fast", conversation_context)
            elif user_query.strip() == '2':
                return self._force_search_with_results("level 2", conversation_context)
            else:  # '3'
                return self._force_search_with_results("24 hour", conversation_context)
        
        # Direct search words - FORCE SEARCH
        elif any(word in query_lower for word in ['closest', 'nearest', 'show me', 'find me']):
            print("🎯 Direct search request - executing search")
            return self._force_search_with_results("general", conversation_context)
        
        # Charging type requests - FORCE SEARCH
        elif any(word in query_lower for word in ['fast charging', 'dc fast', 'quick']):
            print("⚡ Fast charging request - executing search")
            return self._force_search_with_results("dc fast", conversation_context)
        
        # Default to templates
        else:
            print("📋 Using template response")
            return self._fallback_template_response(user_query, conversation_context)

    def _force_search_with_results(self, search_type: str, conversation_context: Dict) -> Dict:
        """Force search and ensure results are shown."""
        try:
            print(f"🔍 Executing {search_type} search...")
            
            # Get location
            lat = conversation_context.get('location', {}).get('lat', 40.7128)
            lon = conversation_context.get('location', {}).get('lon', -74.0060)
            
            if search_type == "dc fast":
                # Find DC Fast stations
                if 'latitude' in self.df.columns:
                    results = self.find_nearby_stations(lat, lon, radius_km=50, limit=10)
                else:
                    results = self.df.copy()
                
                # Filter for DC Fast
                dc_col = pd.to_numeric(results['EV DC Fast Count'], errors='coerce').fillna(0)
                filtered_results = results[dc_col > 0]
                
                if filtered_results.empty:
                    # Show any stations if no DC Fast found
                    filtered_results = results.head(3)
                    response_text = self._format_fallback_stations(filtered_results, "DC Fast")
                else:
                    response_text = self._format_station_results(filtered_results, "DC Fast")
            
            elif search_type == "level 2":
                # Find Level 2 stations
                if 'latitude' in self.df.columns:
                    results = self.find_nearby_stations(lat, lon, radius_km=30, limit=10)
                else:
                    results = self.df.copy()
                
                l2_col = pd.to_numeric(results['EV Level2 EVSE Num'], errors='coerce').fillna(0)
                filtered_results = results[l2_col > 0]
                
                if filtered_results.empty:
                    filtered_results = results.head(3)
                    response_text = self._format_fallback_stations(filtered_results, "Level 2")
                else:
                    response_text = self._format_station_results(filtered_results, "Level 2")
            
            elif search_type == "24 hour":
                # Find 24-hour stations
                filtered_results = self.find_24_hour_stations(limit=5)
                
                if filtered_results.empty:
                    filtered_results = self.df.head(3)
                    response_text = self._format_fallback_stations(filtered_results, "24-hour")
                else:
                    response_text = self._format_station_results(filtered_results, "24-hour")
            
            else:  # general
                # Show any nearby stations
                if 'latitude' in self.df.columns:
                    filtered_results = self.find_nearby_stations(lat, lon, radius_km=25, limit=5)
                else:
                    filtered_results = self.df.head(5)
                
                response_text = self._format_station_results(filtered_results, "nearby")
            
            print(f"✅ Found {len(filtered_results)} stations")
            
            return {
                'response': response_text,
                'context': conversation_context,
                'actions': {'type': 'search_completed'},
                'data': filtered_results.to_dict('records') if not filtered_results.empty else []
            }
        
        except Exception as e:
            print(f"❌ Search error: {e}")
            return {
                'response': f"❌ Sorry, search failed: {str(e)}. Here's what I can help with instead:\n\n" + 
                        self._simple_template_decoder("help", pd.DataFrame(), 'fallback'),
                'context': conversation_context,
                'actions': {'type': 'search_error'}
            }

    def _format_station_results(self, stations_df: pd.DataFrame, search_type: str) -> str:
        """Format stations into a nice display."""
        if stations_df.empty:
            return "❌ No stations found. Try a different search."
        
        result_text = f"⚡ **Found {len(stations_df)} {search_type} charging stations!**\n\n"
        
        for idx, (_, station) in enumerate(stations_df.head(5).iterrows(), 1):
            result_text += f"**{idx}. {station['Station Name']}**\n"
            result_text += f"📍 {station['Street Address']}, {station['City']}\n"
            result_text += f"🕒 {station['Access Days Time']}\n"
            
            # Charging info
            charging_info = []
            if pd.notna(station.get('EV DC Fast Count')) and station['EV DC Fast Count'] > 0:
                charging_info.append(f"⚡ {station['EV DC Fast Count']} DC Fast")
            if pd.notna(station.get('EV Level2 EVSE Num')) and station['EV Level2 EVSE Num'] > 0:
                charging_info.append(f"🔋 {station['EV Level2 EVSE Num']} Level 2")
            if pd.notna(station.get('EV Level1 EVSE Num')) and station['EV Level1 EVSE Num'] > 0:
                charging_info.append(f"🏠 {station['EV Level1 EVSE Num']} Level 1")
            
            if charging_info:
                result_text += f"🔌 {' | '.join(charging_info)}\n"
            
            if 'distance_km' in station and pd.notna(station['distance_km']):
                result_text += f"🚗 {station['distance_km']:.1f} km away\n"
            
            result_text += "\n"
        
        result_text += "**Need directions to any of these stations? Just ask!** 🗺️"
        return result_text

    def _format_fallback_stations(self, stations_df: pd.DataFrame, requested_type: str) -> str:
        """Format fallback stations when specific type not found."""
        if stations_df.empty:
            return f"❌ No {requested_type} stations found in the dataset."
        
        result_text = f"🔍 **No {requested_type} stations found, but here are nearby options:**\n\n"
        
        station = stations_df.iloc[0]
        result_text += f"**📍 {station['Station Name']}**\n"
        result_text += f"Address: {station['Street Address']}, {station['City']}\n"
        result_text += f"Hours: {station['Access Days Time']}\n"
        result_text += f"Contact them for specific charging details.\n\n"
        result_text += "**Try:**\n• Expanding search radius\n• Different charging type\n• Searching by city name"
        
        return result_text

    def _execute_search(self, charger_type: str, context: Dict) -> Dict:
        """Execute search based on LLM decision."""
        lat, lon = context['location']['lat'], context['location']['lon']
        
        try:
            if charger_type == 'dc_fast':
                results = self.find_nearby_stations(lat, lon, radius_km=25, limit=5)
                dc_fast_col = pd.to_numeric(results['EV DC Fast Count'], errors='coerce').fillna(0)
                filtered_results = results[dc_fast_col > 0]
            elif charger_type == 'level2':
                results = self.find_nearby_stations(lat, lon, radius_km=15, limit=5)
                level2_col = pd.to_numeric(results['EV Level2 EVSE Num'], errors='coerce').fillna(0)
                filtered_results = results[level2_col > 0]
            else:  # level1
                results = self.find_nearby_stations(lat, lon, radius_km=10, limit=5)
                level1_col = pd.to_numeric(results['EV Level1 EVSE Num'], errors='coerce').fillna(0)
                filtered_results = results[level1_col > 0]
            
            return {
                'type': 'search_completed',
                'search_results': filtered_results.to_dict('records') if not filtered_results.empty else [],
                'charger_type': charger_type
            }
        except Exception as e:
            return {'type': 'search_error', 'error': str(e)}
        

    def _execute_actual_search(self, query: str, conversation_context: Dict) -> Dict:
        """Actually perform the search with better error handling."""
        query_lower = query.lower()
        
        # Get user location (default to NYC)
        lat = conversation_context.get('location', {}).get('lat', 40.7128)
        lon = conversation_context.get('location', {}).get('lon', -74.0060)
        
        try:
            print(f"🔍 Searching for: {query}")  # Debug
            
            # Check if coordinates exist in dataset
            has_coords = 'latitude' in self.df.columns and 'longitude' in self.df.columns
            
            if any(word in query_lower for word in ['dc', 'fast', 'quick', 'urgent', '1']):
                # Search for DC Fast stations
                print("⚡ Looking for DC Fast stations...")
                
                if has_coords:
                    results = self.find_nearby_stations(lat, lon, radius_km=50, limit=10)  # Increased radius
                else:
                    results = self.df.copy()  # Use all stations if no coordinates
                
                # Better filtering for DC Fast
                dc_fast_col = pd.to_numeric(results['EV DC Fast Count'], errors='coerce').fillna(0)
                filtered_results = results[dc_fast_col > 0]
                
                print(f"Found {len(filtered_results)} DC Fast stations")  # Debug
                search_type = "fast_charging"
                
            elif any(word in query_lower for word in ['level 2', 'l2', 'standard', '2']):
                # Search for Level 2 stations
                print("🔋 Looking for Level 2 stations...")
                
                if has_coords:
                    results = self.find_nearby_stations(lat, lon, radius_km=30, limit=10)
                else:
                    results = self.df.copy()
                
                l2_col = pd.to_numeric(results['EV Level2 EVSE Num'], errors='coerce').fillna(0)
                filtered_results = results[l2_col > 0]
                
                print(f"Found {len(filtered_results)} Level 2 stations")  # Debug
                search_type = "level2_charging"
                
            elif any(word in query_lower for word in ['24', 'hour', 'always', 'open', '3']):
                # Search for 24-hour stations
                print("🌙 Looking for 24-hour stations...")
                filtered_results = self.find_24_hour_stations(limit=10)
                print(f"Found {len(filtered_results)} 24-hour stations")  # Debug
                search_type = "24_hour"
                
            else:
                # Default: show any stations with any charging capability
                print("🔌 Looking for any charging stations...")
                
                if has_coords:
                    filtered_results = self.find_nearby_stations(lat, lon, radius_km=25, limit=10)
                else:
                    # Show first 10 stations if no coordinates
                    filtered_results = self.df.head(10)
                
                print(f"Found {len(filtered_results)} stations")  # Debug
                search_type = "general"
            
            # Generate response with actual results
            if not filtered_results.empty:
                response_text = self._simple_template_decoder(query, filtered_results, search_type)
                return {
                    'response': response_text,
                    'context': conversation_context,
                    'actions': {'type': 'search_completed', 'results_count': len(filtered_results)},
                    'data': filtered_results.to_dict('records')
                }
            else:
                # If no results found, show general stations
                print("❌ No specific results, showing general stations...")
                fallback_results = self.df.head(5)  # Show any 5 stations
                
                response_text = f"""🔍 **No exact matches found, but here are nearby charging options:**

    **🏆 Available Station:** {fallback_results.iloc[0]['Station Name']}
    📍 **Address:** {fallback_results.iloc[0]['Street Address']}, {fallback_results.iloc[0]['City']}
    🕒 **Hours:** {fallback_results.iloc[0]['Access Days Time']}

    **All charging types available - contact station for specific details.**

    **Try these options:**
    • Expand search radius
    • Look for different charging types  
    • Search by city name

    Would you like me to search differently? 🔄"""

                return {
                    'response': response_text,
                    'context': conversation_context,
                    'actions': {'type': 'fallback_results', 'results_count': len(fallback_results)},
                    'data': fallback_results.to_dict('records')
                }
                
        except Exception as e:
            print(f"❌ Search error: {e}")  # Debug
            return {
                'response': f"❌ Sorry, I encountered an error while searching: {str(e)}. Please try again with a different search.",
                'context': conversation_context,
                'actions': {'type': 'search_error'}
            }