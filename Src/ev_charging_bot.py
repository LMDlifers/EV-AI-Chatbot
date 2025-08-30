# imports
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
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
                decoder_model: str = "gpt2", use_decoder: bool = True):
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
    
    def setup_geographic_clustering(self, n_clusters: int = 20) -> None:
        """Setup geographic clustering for location-based optimization."""
        if 'latitude' in self.df.columns and 'longitude' in self.df.columns:
            # Remove rows with missing coordinates
            valid_coords = self.df.dropna(subset=['latitude', 'longitude'])
            
            if len(valid_coords) > 0:
                kmeans = KMeans(n_clusters=min(n_clusters, len(valid_coords)), random_state=42)
                coords = valid_coords[['latitude', 'longitude']].values
                cluster_labels = kmeans.fit_predict(coords)
                
                # Map clusters back to original dataframe
                self.df['geo_cluster'] = -1  # Default for missing coordinates
                self.df.loc[valid_coords.index, 'geo_cluster'] = cluster_labels
                
                self.geo_clusters = kmeans
                print(f"Geographic clustering completed with {n_clusters} clusters")
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
    
    # === CHAT METHODS WITH NATURAL LANGUAGE RESPONSES ===
    
    def chat_semantic_search(self, query: str, limit: int = 3) -> str:
        """
        Perform semantic search and generate a natural language response.
        """
        # Get search results
        results_df = self.semantic_search(query, limit)
        
        # Generate natural language response
        if self.use_decoder and self.decoder_model is not None:
            response = self._generate_response(query, results_df, search_type="semantic")
        else:
            response = self._simple_template_decoder(query, results_df, search_type="semantic")
        
        return response
    
    def chat_find_nearby(self, lat: float, lon: float, radius_km: float = 10, limit: int = 3) -> str:
        """
        Find nearby stations and generate directions/response.
        """
        results_df = self.find_nearby_stations(lat, lon, radius_km, limit)
        
        # Create location context
        location_query = f"charging stations near coordinates {lat}, {lon} within {radius_km}km"
        
        if self.use_decoder and self.decoder_model is not None:
            response = self._generate_response(location_query, results_df, search_type="location")
        else:
            response = self._simple_template_decoder(location_query, results_df, search_type="location")
        
        return response
    
    def chat_find_fast_charging(self, city: str = None, limit: int = 3) -> str:
        """
        Find fast charging stations with natural language response.
        """
        results_df = self.find_fast_charging(city, limit)
        
        city_text = f"in {city}" if city else ""
        query = f"fast charging stations {city_text}"
        
        if self.use_decoder and self.decoder_model is not None:
            response = self._generate_response(query, results_df, search_type="fast_charging")
        else:
            response = self._simple_template_decoder(query, results_df, search_type="fast_charging")
        
        return response
    
    def chat_find_level2_charging(self, city: str = None, min_ports: int = 1, limit: int = 3) -> str:
        """
        Find Level 2 charging stations with natural language response.
        """
        results_df = self.find_level2_charging(city, min_ports, limit)
        
        city_text = f"in {city}" if city else ""
        query = f"Level 2 charging stations {city_text}"
        
        if self.use_decoder and self.decoder_model is not None:
            response = self._generate_response(query, results_df, search_type="level2_charging")
        else:
            response = self._simple_template_decoder(query, results_df, search_type="level2_charging")
        
        return response
    
    # === RESPONSE GENERATION METHODS ===
    
    def _generate_response(self, query: str, results_df: pd.DataFrame, search_type: str) -> str:
        """
        Generate natural language response using decoder model.
        """
        # Prepare context from search results
        context = self._prepare_context(results_df, search_type)
        
        # Create prompt for the decoder
        prompt = self._create_prompt(query, context, search_type)
        
        # Generate response using decoder
        response = self._decode_response(prompt)
        
        return response
    
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
    
    def _create_prompt(self, query: str, context: str, search_type: str) -> str:
        """
        Create a structured prompt for the decoder model.
        """

        prompt_templates = {
            "semantic": 
            
                f"""User asked: "{query}"

                Here are the relevant charging stations I found:
                {context}

                Please provide a helpful response with directions and recommendations:""",
                        
            "location": 

                f"""User is looking for: {query}

                Here are the nearby charging stations:
                {context}

                Please provide directions and recommendations:""",
                        
            "fast_charging": 

                f"""User is looking for: {query}

                Here are the available fast charging options:
                {context}

                Please provide helpful information and directions:""",
                        
            "level2_charging": 

                f"""User is looking for: {query}

                Here are the available Level 2 charging options:
                {context}

                Please provide helpful information and recommendations:"""
        }
                
        return prompt_templates.get(search_type, prompt_templates["semantic"])
    
    def _decode_response(self, prompt: str, max_length: int = 200, temperature: float = 0.7) -> str:
        """
        Generate response using the decoder model.
        """
        # Encode the prompt
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
        
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
    
    def _simple_template_decoder(self, query: str, results_df: pd.DataFrame, search_type: str) -> str:
        """
        Template-based response generation (lighter alternative).
        """
        if results_df.empty:
            return f"I couldn't find any charging stations matching '{query}'. Try broadening your search criteria."
        
        station_count = len(results_df)
        first_station = results_df.iloc[0]
        
        responses = {
            "semantic": f"I found {station_count} stations matching '{query}'. The top result is {first_station['Station Name']} located at {first_station['Street Address']}, {first_station['City']}. They're open {first_station['Access Days Time']}. Would you like directions or more details about any of these stations?",
            
            "location": f"There are {station_count} charging stations within your specified area. The closest is {first_station['Station Name']} at {first_station['Street Address']}, {first_station['City']}. You can charge there during {first_station['Access Days Time']}. Shall I provide turn-by-turn directions?",
            
            "fast_charging": f"Great! I found {station_count} fast charging stations. {first_station['Station Name']} in {first_station['City']} has {first_station['EV DC Fast Count']} DC fast charging ports and is available {first_station['Access Days Time']}. This will get you charged up quickly!",
            
            "level2_charging": f"I found {station_count} Level 2 charging stations. {first_station['Station Name']} in {first_station['City']} has {first_station['EV Level2 EVSE Num']} Level 2 ports and is available {first_station['Access Days Time']}. Perfect for longer charging sessions!"
        }
        
        return responses.get(search_type, responses["semantic"])
    
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
    