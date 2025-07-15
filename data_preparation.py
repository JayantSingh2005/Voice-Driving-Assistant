# J:\Projects\OK Driver\data_preparation.py
import sys
import os
import json
import re # For parsing semantic parses
### --- FIX START --- ###
# Import the string module to handle punctuation robustly.
import string
### --- FIX END --- ###
from typing import List, Dict, Any, Optional
from collections import defaultdict
from datasets import load_dataset 

# Get the absolute path of the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Ensure this directory is at the beginning of the Python import path
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Now, import Schema.py, which should be directly findable
from Schema import CanonicalSchema

class DatasetProcessor:
    def _init_(self, output_dir: str = None):
        """
        Initializes the DatasetProcessor with schema and output directory.
        Args:
            output_dir (str): Directory where processed data will be saved.
        """
        self.schema = CanonicalSchema()
        self.canonical_intents = self.schema.get_all_intents()
        self.canonical_entities = self.schema.get_all_entities()
        self.canonical_emotion_labels = self.schema.get_all_emotion_labels()
        self.output_dir = output_dir if output_dir else "prepared_data"
        os.makedirs(self.output_dir, exist_ok=True)
        self.massive_to_canonical_intent_map: Dict[str, str] = self._build_massive_intent_map()
        self.massive_to_canonical_entity_map: Dict[str, str] = self._build_massive_entity_map()
        
        # NEW: Mappings for Hinglish TOP dataset
        self.top_to_canonical_intent_map: Dict[str, str] = self._build_top_intent_map()
        self.top_to_canonical_entity_map: Dict[str, str] = self._build_top_entity_map()

        self.entity_to_id = {}
        current_id = 0
        for entity in sorted(list(self.canonical_entities)):
            self.entity_to_id[f"B-{entity}"] = current_id
            current_id += 1
            self.entity_to_id[f"I-{entity}"] = current_id
            current_id += 1
        self.entity_to_id['O'] = current_id 
        self.id_to_entity = {i: entity for entity, i in self.entity_to_id.items()}
        
        sorted_canonical_intents = sorted(list(self.canonical_intents))
        self.intent_to_id = {intent: i for i, intent in enumerate(sorted_canonical_intents)}
        self.id_to_intent = {i: intent for intent, i in self.intent_to_id.items()}

        sorted_canonical_emotions = sorted(list(self.canonical_emotion_labels))
        self.emotion_to_id = {emotion: i for i, emotion in enumerate(sorted_canonical_emotions)}
        self.id_to_emotion = {i: emotion for emotion, i in self.emotion_to_id.items()}

        print("Initialized DatasetProcessor.")
        print(f"Total Canonical Intents: {len(self.canonical_intents)}")
        print(f"Total Canonical Entities (BIO tags including O): {len(self.entity_to_id)}")
        print(f"Total Canonical Emotions: {len(self.canonical_emotion_labels)}")

    def _build_massive_intent_map(self) -> Dict[str, str]:
        """
        Builds the mapping from MASSIVE dataset intents (scenario_intent) to your canonical intents.
        Any MASSIVE intent not explicitly mapped will default to "unsupported_query".
        """
        intent_map = defaultdict(lambda: "unsupported_query")
        # I & II. Dashcam & Road Safety Intents:
        # These are custom intents, so MASSIVE doesn't directly map to them.
        # MASSIVE data for these might end up as "unsupported_query" or you manually map.
        # Example: if 'general_camera_on' existed in MASSIVE, map to 'dashcam_record_on'.
        # For now, assuming these are mainly from your custom car-domain data if you add it.
        # III. Navigation (Aligned with MASSIVE)
        intent_map["navigation_set_destination"] = "nav_start_navigation"
        intent_map["navigation_get_distance"] = "unsupported_query" # Mapped to unsupported as 'nav_query_distance' is not in schema.
        intent_map["navigation_query_distance"] = "unsupported_query"
        intent_map["navigation_get_directions"] = "nav_start_navigation" # Treat asking for directions as starting navigation
        intent_map["navigation_stop_navigation"] = "nav_stop_navigation"
        intent_map["navigation_get_location_current"] = "nav_query_current_location"
        intent_map["navigation_query_poi"] = "nav_find_nearest"
        intent_map["navigation_get_eta"] = "nav_query_eta"
        intent_map["navigation_query_eta"] = "nav_query_eta"
        intent_map["navigation_get_traffic"] = "nav_query_traffic"
        intent_map["navigation_query_traffic"] = "nav_query_traffic"
        intent_map["navigation_reroute"] = "nav_reroute"
        # IV. Communication (Aligned with MASSIVE)
        intent_map["messaging_send_message"] = "comm_send_message"
        intent_map["calling_make_call"] = "comm_make_call"
        intent_map["messaging_read_messages"] = "comm_read_message"
        intent_map["messaging_reply_to_message"] = "comm_reply_message"
        intent_map["app_open"] = "comm_open_app"
        # V. Information & Query (Aligned with MASSIVE, some custom)
        intent_map["datetime_query"] = "query_date" # General datetime query can default to date
        intent_map["datetime_get_time"] = "query_time"
        intent_map["datetime_get_date"] = "query_date"
        intent_map["weather_get_forecast"] = "query_weather"
        intent_map["weather_query"] = "query_weather"
        intent_map["alarm_query"] = "reminder_query" 
        intent_map["alarm_set_alarm"] = "reminder" 
        intent_map["general_query"] = "unsupported_query" # Too generic, better to use custom data for specific queries
        # VI. General Utility & Settings (Aligned with MASSIVE, some custom)
        intent_map["audio_volume_up"] = "app_adjust_volume"
        intent_map["audio_volume_down"] = "app_adjust_volume"
        intent_map["audio_set_volume"] = "app_adjust_volume"
        intent_map["play_music"] = "music_play"
        intent_map["music_play"] = "music_play"
        intent_map["music_pause"] = "music_pause"
        intent_map["music_resume"] = "music_play"
        intent_map["music_stop"] = "music_pause"
        intent_map["audio_volume_mute"] = "app_mute"
        intent_map["audio_volume_unmute"] = "app_unmute"
        intent_map["app_change_language"] = "app_change_language"
        intent_map["app_settings"] = "app_open_settings"
        intent_map["app_close"] = "app_close_app"
        # VII. Conversational & System Management (Aligned with MASSIVE, some custom)
        intent_map["general_greeting"] = "general_greet"
        intent_map["general_goodbye"] = "general_farewell"
        intent_map["general_affirm"] = "general_confirm"
        intent_map["general_negate"] = "general_deny"
        intent_map["general_thank_you"] = "general_thank_you"
        intent_map["general_joke"] = "unsupported_query" # Not relevant for car assistant
        intent_map["general_what_can_i_do"] = "general_help"
        intent_map["general_cancel"] = "general_cancel"
        intent_map["general_start_over"] = "general_start_over"
        # Other potential MASSIVE intents that are out of scope for a car assistant AI:
        intent_map["iot_control_lights_turn_on"] = "unsupported_query" 
        intent_map["iot_control_lights_turn_off"] = "unsupported_query"
        intent_map["cooking_set_timer"] = "unsupported_query"
        intent_map["news_query"] = "unsupported_query"
        intent_map["calendar_query"] = "unsupported_query"
        intent_map["alarm_cancel"] = "unsupported_query" # Handled by reminder_delete if added to schema
        intent_map["alarm_snooze"] = "unsupported_query" # Handled by snooze_alarm if added to schema
        # NEW MAPPINGS based on the latest report from mapping_discovery_script.py:
        intent_map["16_48"] = "reminder" # "wake me up at nine am on friday"
        intent_map["10_46"] = "app_mute" # "olly quiet"
        intent_map["8_1"] = "unsupported_query" # IoT: "make the lighting bit more warm here"
        intent_map["8_40"] = "unsupported_query" # IoT: "time to sleep" (implies home automation)
        intent_map["8_31"] = "unsupported_query" # IoT: "olly dim the lights in the hall"
        intent_map["8_34"] = "unsupported_query" # IoT: "olly clean the flat"
        intent_map["2_32"] = "unsupported_query" # Calendar/Event: "check when the show starts"
        intent_map["3_45"] = "music_play" # "i want to listen arijit singh song once again"
        intent_map["9_12"] = "unsupported_query" # General query: "check my car is ready"
        intent_map["9_5"] = "general_greet" # "what's up"
        intent_map["5_0"] = "query_time" # "tell me the time in moscow"
        intent_map["5_38"] = "query_time" # "tell me the time in g.m. t. plus five"
        intent_map["14_3"] = "unsupported_query" # Ordering food: "olly list most rated delivery options for chinese food"
        intent_map["16_52"] = "reminder_delete" # "stop seven am alarm" (Requires reminder_delete in Schema.py)
        intent_map["16_23"] = "reminder_query" # "please list active alarms"
        intent_map["4_22"] = "unsupported_query" # News/Sports: "what's happening in football today"
        intent_map["15_43"] = "unsupported_query" # Music preference: "i like rock music"
        intent_map["15_57"] = "unsupported_query" # Music query: "who's current music's author"
        intent_map["8_18"] = "unsupported_query" # IoT: "make lights brightener"
        intent_map["14_16"] = "unsupported_query" # Ordering food: "please order some sushi for dinner"
        intent_map["17_13"] = "query_weather" # "is it raining"
        intent_map["15_28"] = "music_shuffle_on" # "shuffle this playlist" (Requires music_shuffle_on in Schema.py)
        intent_map["9_25"] = "unsupported_query" # General entertainment: "make me laugh"
        intent_map["15_7"] = "unsupported_query" # Music dislike: "i don't like it" (Consider adding music_dislike_track to schema if desired)
        intent_map["10_29"] = "app_adjust_volume" # "change the volume"
        intent_map["8_56"] = "unsupported_query" # IoT: "make me a coffee"
        intent_map["10_14"] = "app_adjust_volume" # "turn up the volume"
        intent_map["8_24"] = "unsupported_query" # IoT: "turn on the plug"
        intent_map["8_41"] = "unsupported_query" # IoT: "and the light began"
        intent_map["8_8"] = "unsupported_query" # IoT: "turn off the rice cooker socket"
        intent_map["10_35"] = "app_adjust_volume" # "you're too loud"
        intent_map["12_4"] = "unsupported_query" # Finance/News: "keep me updated on stock market prices"
        intent_map["3_36"] = "music_play" # "play my favorite pandora station"
        intent_map["6_19"] = "nav_find_nearest" # "find me a nice restaurant for dinner" (POI)
        intent_map["12_49"] = "unsupported_query" # General query: "what is the gross domestic product of us"
        intent_map["2_50"] = "reminder" # "remind me if anything else happens"
        intent_map["3_20"] = "unsupported_query" # Audio/Book: "please repeat the last sentence from that book"
        intent_map["3_58"] = "music_previous_track" # "go back" (in music context) (Requires music_previous_track in Schema.py)
        intent_map["0_27"] = "unsupported_query" # Social Media: "what is going on right now on twitter"
        intent_map["1_42"] = "nav_reroute" # "you need to give me different directions"
        intent_map["7_33"] = "comm_send_message" # "i did not want you to send that text yet wait until i say send"
        intent_map["6_55"] = "unsupported_query" # Entertainment query: "should i watch this movie"
        intent_map["11_59"] = "unsupported_query" # To-do list management: "i finished my to do list"
        intent_map["3_51"] = "unsupported_query" # General entertainment: "games"
        intent_map["1_2"] = "unsupported_query" # Travel/Flights: "cheapest flights to kansas city from dallas"
        intent_map["6_6"] = "general_help" # "what is happening olly"
        intent_map["7_44"] = "comm_read_message" # "emails"
        intent_map["1_11"] = "nav_query_traffic" # "just how bad is traffic on my commute"
        intent_map["13_37"] = "unsupported_query" # Cooking: "what to cook for lunch"
        intent_map["12_26"] = "unsupported_query" # General query: "what is skynet"
        intent_map["2_30"] = "unsupported_query" # Calendar/Event management: "olly cancel business meeting on wednesday"
        intent_map["11_53"] = "unsupported_query" # To-do list management: "remove grocery shopping from my to do list on sunday"
        intent_map["13_9"] = "unsupported_query" # Cooking: "how long should i boil the eggs"
        intent_map["7_17"] = "comm_make_call" # "call mom"
        intent_map["11_21"] = "unsupported_query" # To-do list management: "create a new to do list"
        intent_map["1_54"] = "unsupported_query" # Ride-sharing/Taxi: "olly book me a taxi to leith in half an hour"
        intent_map["12_39"] = "unsupported_query" # Calculator: "ten percentages of hundred"
        intent_map["0_47"] = "comm_send_message" # "send a message on facebook to pending friend requests that my friend list is full and i will be creating a new profile soon"
        intent_map["12_10"] = "unsupported_query" # Finance: "alexa i would like my to tell me the trend on foreign exchange rates"
        intent_map["7_15"] = "unsupported_query" # Email specific: "add new email to anna"
        return intent_map

    def _build_massive_entity_map(self) -> Dict[str, str]:
        """
        Builds the mapping from MASSIVE dataset entity names to your canonical entity names.
        Entities not present in schema.py have been explicitly excluded or will be ignored.
        """
        entity_map = {
            # Communication
            "name": "contact_name",
            "contact_name": "contact_name",
            "person_name": "contact_name",
            "recipient": "contact_name", 
            "message": "message_body",
            "app_name": "app_name",
            
            # Music
            "song": "song_name",
            "song_name": "song_name",
            "artist_name": "artist_name",
            "artist": "artist_name",
            
            # Navigation
            "place_of_interest": "location_name",
            "business_name": "location_name",
            "street_name": "destination_address",
            "city": "location_name",
            "state": "location_name",
            "country": "location_name",
            "geographic_poi": "location_name",
            "poi_category": "poi_category",
            "address": "destination_address",
            "traffic_condition": "traffic_condition",
            
            # Datetime
            "time": "datetime",
            "date": "datetime",
            "date_time": "datetime",
            "timeofday": "datetime",
            
            # Reminders (MASSIVE's alarm/timer names mapped to our 'task')
            "alarm_name": "task",
            "timer_name": "task",
            
            # General Query-related
            "query": "specific_query", 
            
            # General Utility & Settings
            "volume_level": "volume_percentage",
            "direction": "volume_direction", 
            "language": "language_name",
            "device": "device_component", 
        }
        return entity_map

    # NEW: Mappings for Hinglish TOP Dataset
    def _build_top_intent_map(self) -> Dict[str, str]:
        """
        Builds the mapping from Hinglish TOP dataset intents to your canonical intents.
        Add more mappings as needed based on the TOP dataset's actual intents.
        """
        intent_map = defaultdict(lambda: "unsupported_query")
        
        # Example mappings (adjust these based on actual TOP dataset intents)
        # You'll need to inspect the 'cs_parse' column to see the exact IN: values.
        intent_map["IN:PlayMusic"] = "music_play"
        intent_map["IN:PauseMusic"] = "music_pause"
        intent_map["IN:SetAlarm"] = "reminder" # This was 'reminder' previously, might need to be 'alarm_set_alarm' if you add it.
        intent_map["IN:GetWeather"] = "query_weather"
        intent_map["IN:SetReminder"] = "reminder"
        intent_map["IN:MakeCall"] = "comm_make_call"
        intent_map["IN:SendMessage"] = "comm_send_message"
        intent_map["IN:GetLocation"] = "nav_query_current_location"
        intent_map["IN:FindPoi"] = "nav_find_nearest"
        intent_map["IN:StartNavigation"] = "nav_start_navigation"
        intent_map["IN:StopNavigation"] = "nav_stop_navigation"
        intent_map["IN:GetETA"] = "nav_query_eta"
        intent_map["IN:GetTraffic"] = "nav_query_traffic"
        intent_map["IN:Reroute"] = "nav_reroute"
        intent_map["IN:ChangeLanguage"] = "app_change_language"
        intent_map["IN:SetVolume"] = "app_adjust_volume"
        intent_map["IN:Mute"] = "app_mute"
        intent_map["IN:Unmute"] = "app_unmute"
        
        # General conversational/utility intents that might appear
        intent_map["IN:Greeting"] = "general_greet"
        intent_map["IN:ThankYou"] = "general_thank_you"
        intent_map["IN:Affirm"] = "general_confirm"
        intent_map["IN:Negate"] = "general_deny"
        intent_map["IN:Goodbye"] = "general_farewell"
        intent_map["IN:Help"] = "general_help"
        intent_map["IN:Cancel"] = "general_cancel"
        intent_map["IN:StartOver"] = "general_start_over"

        # NEW MAPPINGS based on the latest report from mapping_discovery_script.py:
        # Reminders/Alarms
        intent_map["IN:CREATE_ALARM"] = "reminder"
        intent_map["IN:DELETE_REMINDER"] = "reminder_delete" # Ensure "reminder_delete" is in Schema.py
        intent_map["IN:GET_ALARM"] = "reminder_query"
        intent_map["IN:CREATE_TIMER"] = "reminder"
        intent_map["IN:DELETE_ALARM"] = "reminder_delete" # Ensure "reminder_delete" is in Schema.py
        intent_map["IN:RESUME_TIMER"] = "unsupported_query" # Keep as unsupported for car context
        intent_map["IN:GET_TIMER"] = "reminder_query"
        intent_map["IN:RESTART_TIMER"] = "unsupported_query" # Keep as unsupported for car context
        intent_map["IN:UNSUPPORTED_TIMER"] = "unsupported_query"
        intent_map["IN:DELETE_TIMER"] = "reminder_delete" # Ensure "reminder_delete" is in Schema.py
        intent_map["IN:PAUSE_TIMER"] = "unsupported_query" # Keep as unsupported for car context
        intent_map["IN:GET_REMINDER"] = "reminder_query"
        intent_map["IN:GET_ESTIMATED_DEPARTURE"] = "nav_query_eta"
        intent_map["IN:UPDATE_REMINDER"] = "reminder"
        intent_map["IN:UPDATE_REMINDER_TODO"] = "reminder"
        intent_map["IN:ADD_TIME_TIMER"] = "unsupported_query" # Keep as unsupported for car context
        intent_map["IN:SUBTRACT_TIME_TIMER"] = "unsupported_query" # Keep as unsupported for car context
        intent_map["IN:UPDATE_ALARM"] = "reminder"
        intent_map["IN:SILENCE_ALARM"] = "app_mute" # Ensure "app_mute" is in Schema.py
        intent_map["IN:UPDATE_TIMER"] = "unsupported_query" # Keep as unsupported for car context
        intent_map["IN:SNOOZE_ALARM"] = "unsupported_query" # Keep as unsupported for car context
        intent_map["IN:UPDATE_REMINDER_DATE_TIME"] = "reminder"
        intent_map["IN:GET_REMINDER_LOCATION"] = "reminder_query"
        intent_map["IN:UNSUPPORTED_ALARM"] = "unsupported_query"
        intent_map["IN:GET_REMINDER_DATE_TIME"] = "reminder_query"
        intent_map["IN:GET_REMINDER_AMOUNT"] = "reminder_query"
        # Communication/Messaging
        intent_map["IN:SEND_MESSAGE"] = "comm_send_message" # NOW MAPPED
        intent_map["IN:GET_MESSAGE"] = "comm_read_message"
        intent_map["IN:REACT_MESSAGE"] = "unsupported_query" # Keep as unsupported for car context
        # Navigation
        intent_map["IN:GET_ESTIMATED_ARRIVAL"] = "nav_query_eta"
        intent_map["IN:GET_DISTANCE"] = "unsupported_query"
        intent_map["IN:GET_DIRECTIONS"] = "nav_start_navigation"
        intent_map["IN:UNSUPPORTED_NAVIGATION"] = "unsupported_query"
        intent_map["IN:GET_INFO_ROAD_CONDITION"] = "nav_query_traffic"
        intent_map["IN:UPDATE_DIRECTIONS"] = "nav_reroute"
        intent_map["IN:GET_ESTIMATED_DURATION"] = "nav_query_eta" # NOW MAPPED
        # Weather
        intent_map["IN:GET_WEATHER"] = "query_weather" # NOW MAPPED
        intent_map["IN:UNSUPPORTED_WEATHER"] = "unsupported_query"
        intent_map["IN:GET_SUNSET"] = "query_weather"
        intent_map["IN:GET_SUNRISE"] = "query_weather"
        intent_map["IN:GET_INFO_TRAFFIC"] = "nav_query_traffic" # NOW MAPPED
        # Music
        intent_map["IN:REPLAY_MUSIC"] = "music_play"
        intent_map["IN:CREATE_PLAYLIST_MUSIC"] = "music_play" # If you add "music_create_playlist" to schema, map to that.
        intent_map["IN:PLAY_MUSIC"] = "music_play" # NOW MAPPED
        intent_map["IN:PAUSE_MUSIC"] = "music_pause" # NOW MAPPED
        intent_map["IN:SKIP_TRACK_MUSIC"] = "music_next_track" # Ensure "music_next_track" is in Schema.py
        intent_map["IN:ADD_TO_PLAYLIST_MUSIC"] = "music_play" # If you add "music_add_to_playlist" to schema, map to that.
        intent_map["IN:PREVIOUS_TRACK_MUSIC"] = "music_previous_track" # Ensure "music_previous_track" is in Schema.py
        intent_map["IN:UNSUPPORTED_MUSIC"] = "unsupported_query"
        intent_map["IN:REMOVE_FROM_PLAYLIST_MUSIC"] = "unsupported_query" # If you add "music_remove_from_playlist" to schema, map to that.
        intent_map["IN:STOP_MUSIC"] = "music_pause"
        intent_map["IN:LIKE_MUSIC"] = "unsupported_query" # If you add "music_like_track" to schema, map to that.
        intent_map["IN:LOOP_MUSIC"] = "music_repeat_on" # Ensure "music_repeat_on" is in Schema.py
        intent_map["IN:START_SHUFFLE_MUSIC"] = "music_shuffle_on" # Ensure "music_shuffle_on" is in Schema.py
        intent_map["IN:DISLIKE_MUSIC"] = "unsupported_query" # If you add "music_dislike_track" to schema, map to that.
        # Events/General Query
        intent_map["IN:GET_EVENT"] = "unsupported_query" # If you add "query_event" to schema, map to that.
        intent_map["IN:UNSUPPORTED_EVENT"] = "unsupported_query"
        intent_map["IN:CREATE_REMINDER"] = "reminder" # NOW MAPPED
        return intent_map

    def _build_top_entity_map(self) -> Dict[str, str]:
        """
        Builds the mapping from Hinglish TOP dataset slot names to your canonical entity names.
        Add more mappings as needed based on the TOP dataset's actual slot names.
        """
        entity_map = {
            # Music
            "SL:ArtistName": "artist_name",
            "SL:SongName": "song_name",
            "SL:PlaylistName": "playlist_name", # Assuming you have this in your schema, if not, map to general_query or unsupported
            "SL:AlbumName": "album_name", # Assuming you have this
            
            # Navigation
            "SL:Location": "location_name",
            "SL:PoiCategory": "poi_category",
            "SL:Destination": "destination_address",
            "SL:TrafficCondition": "traffic_condition",
            
            # Communication
            "SL:ContactName": "contact_name",
            "SL:MessageBody": "message_body",
            "SL:AppName": "app_name",

            # Datetime / Reminders
            "SL:Time": "datetime",
            "SL:Date": "datetime",
            "SL:DateTime": "datetime",
            "SL:Task": "task",
            
            # General / Settings
            "SL:VolumeLevel": "volume_percentage",
            "SL:Direction": "volume_direction", 
            "SL:Language": "language_name",
            "SL:DeviceComponent": "device_component"
            
            # You'll likely need to add more mappings here as you encounter TOP slots
            # that match your schema.
        }
        return entity_map

    def _map_goemotions_to_canonical(self, go_emotions_labels: List[int]) -> str:
        """
        Helper function to map GoEmotions multi-label output (27 emotions) to a single
        of your 7 canonical emotions. This is a simplified mapping.
        GoEmotions labels (from their documentation, common IDs):
        0: admiration, 1: amusement, 2: anger, 3: annoyance, 4: approval, 5: caring,
        6: confusion, 7: desire, 8: disappointment, 9: disapproval, 10: disgust,
        11: embarrassment, 12: excitement, 13: fear, 14: gratitude, 15: grief,
        16: joy, 17: love, 18: nervousness, 19: optimism, 20: pride, 21: realization,
        22: relief, 23: remorse, 24: sadness, 25: surprise, 26: neutral
        """
        # Prioritize mapping to your 6 specific emotions, then default to neutral
        
        # Joy
        if 16 in go_emotions_labels or 1 in go_emotions_labels or 12 in go_emotions_labels or 14 in go_emotions_labels or 17 in go_emotions_labels or 19 in go_emotions_labels or 20 in go_emotions_labels or 22 in go_emotions_labels:
            return "joy"
        # Anger
        if 2 in go_emotions_labels or 3 in go_emotions_labels or 9 in go_emotions_labels:
            return "anger"
        # Sadness
        if 24 in go_emotions_labels or 8 in go_emotions_labels or 15 in go_emotions_labels or 23 in go_emotions_labels:
            return "sadness"
        # Fear
        if 13 in go_emotions_labels or 18 in go_emotions_labels:
            return "fear"
        # Surprise
        if 25 in go_emotions_labels or 6 in go_emotions_labels: # Confusion sometimes overlaps with mild surprise
            return "surprise"
        # Disgust
        if 10 in go_emotions_labels:
            return "disgust"
        
        # Default to neutral if no specific emotion is found or if it's explicitly neutral
        return "neutral" 

    def _parse_top_semantic_parse(self, semantic_parse: str, utterance: str) -> Dict[str, Any]:
        """
        Parses a Hinglish TOP dataset semantic parse string
        (e.g., "[IN:PlayMusic [SL:ArtistName Bruno Mars]]")
        and extracts intent and entities.
        Returns a dict with 'intent' and 'annotations' (list of dicts like MASSIVE).
        """
        parsed_intent = "unsupported_query" # Default intent
        parsed_annotations = []
        # Extract Intent
        # This regex captures the main intent: [IN:IntentName
        match_main_intent = re.match(r'\[IN:([^\]\s]+)', semantic_parse)
        if match_main_intent:
            top_intent_raw = "IN:" + match_main_intent.group(1)
            parsed_intent = self.top_to_canonical_intent_map.get(top_intent_raw, "unsupported_query")
 
        # Extract Entities/Slots
        # This regex looks for [SL:SlotName Value] or [SL:SlotName Value1 Value2]
        # It's a simplified parser and might need adjustments for complex nested slots.
        slot_matches = re.finditer(r'\[SL:([A-Za-z]+)\s([^\]]+)\]', semantic_parse)
        
        for match in slot_matches:
            top_slot_name = "SL:" + match.group(1)
            slot_value = match.group(2).strip()
            
            canonical_entity_name = self.top_to_canonical_entity_map.get(top_slot_name)
            
            if canonical_entity_name:
                parsed_annotations.append({
                    "name": canonical_entity_name, # Mapped to your canonical entity name
                    "value": slot_value
                })
        
        return {
            "intent": parsed_intent,
            "annotations": parsed_annotations
        }

    def process_data_entry(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Processes a single data entry, now including emotion.
        Args:
            entry (Dict[str, Any]): A single entry from the raw dataset,
                                    expected to have 'utt', 'scenario', 'intent', 'annotations',
                                    'lang', and crucially, 'emotion_label_raw'.
        Returns:
            Optional[Dict[str, Any]]: A dictionary containing processed data, or None if skipped.
        """
        utterance = entry["utt"]
        
        # Determine canonical intent
        # Check if it's from MASSIVE (has scenario/intent) or a custom source (has direct intent)
        if 'scenario' in entry and 'intent' in entry: # Likely from MASSIVE
            massive_intent_key = f"{entry['scenario']}_{entry['intent']}"
            canonical_intent = self.massive_to_canonical_intent_map[massive_intent_key]
        else: # Likely from GoEmotions or Hinglish TOP (where 'intent' is already canonical or parsed)
            canonical_intent = entry.get("intent", "unsupported_query")
            
        if canonical_intent not in self.canonical_intents:
            canonical_intent = "unsupported_query"

        # Determine BIO entity tags
        # Get annotations from 'annotations' (MASSIVE/Hinglish TOP) or 'slot_and_value' (MASSIVE fallback)
        annotations = entry.get("annotations", []) 
        if not annotations:
            annotations = entry.get("slot_and_value", []) 
        
        tokens_for_matching = utterance.lower().split()
        bio_tags = ['O'] * len(tokens_for_matching) 

        ### --- FIX START --- ###
        # This translator is used to strip punctuation for more robust entity matching.
        translator = str.maketrans('', '', string.punctuation)
        # Create a normalized version of the utterance tokens for matching purposes.
        normalized_utterance_tokens = [t.translate(translator) for t in tokens_for_matching]
        ### --- FIX END --- ###

        for annotation in annotations:
            if not isinstance(annotation, dict) or 'name' not in annotation or 'value' not in annotation:
                continue
            
            # 'name' is already canonical for Hinglish TOP after parsing.
            # For MASSIVE, we still need to map 'massive_entity_name'.
            if entry.get('source') == 'Hinglish_TOP': # Custom flag to indicate Hinglish TOP data
                canonical_entity_name = annotation['name'] # Already canonical after _parse_top_semantic_parse
            else:
                massive_entity_name = annotation['name']
                canonical_entity_name = self.massive_to_canonical_entity_map.get(massive_entity_name)
            
            if not canonical_entity_name or canonical_entity_name not in self.canonical_entities:
                continue
            
            ### --- FIX START --- ###
            # This block replaces the original, fragile entity matching logic.
            entity_value_raw = annotation['value']
            entity_value_str = str(entity_value_raw)
            # Tokenize the original entity value to get the correct number of tags (e.g., "New York" is 2 tokens)
            entity_tokens = entity_value_str.lower().split()
            
            # Normalize the entity tokens by stripping punctuation, just like the utterance tokens.
            # Also, remove any empty strings that might result from stripping.
            normalized_entity_tokens = [t.translate(translator) for t in entity_tokens if t.translate(translator)]

            if not normalized_entity_tokens:
                continue # Skip if the entity is empty after normalization (e.g., was just a punctuation mark).

            # Search for the normalized entity sequence within the normalized utterance sequence.
            for i in range(len(normalized_utterance_tokens) - len(normalized_entity_tokens) + 1):
                if normalized_utterance_tokens[i:i+len(normalized_entity_tokens)] == normalized_entity_tokens:
                    # If a match is found, apply the B- and I- tags to the original BIO tag list.
                    # This ensures we tag the correct tokens, even if they had punctuation.
                    bio_tags[i] = "B-" + canonical_entity_name
                    for j in range(1, len(entity_tokens)):
                        bio_tags[i+j] = "I-" + canonical_entity_name
                    break # Found the entity, stop searching for this annotation.
            ### --- FIX END --- ###

        bio_tag_ids = []
        for i, tag in enumerate(bio_tags):
            if tag in self.entity_to_id: 
                bio_tag_ids.append(self.entity_to_id[tag])
            else:
                print(f"WARNING: Unknown BIO tag '{tag}' for token '{tokens_for_matching[i]}' in utterance '{utterance}'. Defaulting to 'O'.")
                bio_tag_ids.append(self.entity_to_id['O'])

        if len(tokens_for_matching) != len(bio_tag_ids):
            print(f"ERROR: Token/Tag ID mismatch for utterance: '{utterance}' (Tokens: {len(tokens_for_matching)}, Tags: {len(bio_tag_ids)}). Skipping entry.")
            return None 

        raw_emotion_from_source = entry.get("emotion_label_raw", "neutral")
        canonical_emotion = raw_emotion_from_source 
        if canonical_emotion not in self.canonical_emotion_labels:
            canonical_emotion = "neutral" 

        return {
            "utterance": utterance,
            "tokens": utterance.split(),
            "intent_label": canonical_intent,
            "intent_label_id": self.intent_to_id[canonical_intent],
            "entity_bio_tags": bio_tags,
            "entity_bio_tag_ids": bio_tag_ids,
            "emotion_label": canonical_emotion,
            "emotion_label_id": self.emotion_to_id[canonical_emotion],
            "language": entry["lang"]
        }

    def _load_massive_data_for_lang(self, split_name: str, lang: str) -> List[Dict[str, Any]]:
        """
        Loads and processes a single language split from the MASSIVE dataset.
        Assigns a default "neutral" emotion label.
        """
        processed_data = []
        try:
            raw_massive_split = load_dataset("AmazonScience/massive", lang, split=split_name)
            print(f"  Loaded {raw_massive_split.num_rows} entries from MASSIVE for '{lang}'.")
            for entry in raw_massive_split:
                massive_entry_with_defaults = {
                    **entry, 
                    'lang': lang, 
                    'emotion_label_raw': "neutral",
                    'source': 'MASSIVE' # Add source for tracking
                }
                processed_entry = self.process_data_entry(massive_entry_with_defaults)
                if processed_entry:
                    processed_data.append(processed_entry)
        except Exception as e:
            print(f"ERROR: Could not load MASSIVE data for language '{lang}' in split '{split_name}'. Exception: {e}")
        return processed_data

    def _load_go_emotions_data(self, split_name: str, language: str = "en-US") -> List[Dict[str, Any]]:
        """
        Loads and processes data from the GoEmotions dataset for emotion classification.
        Assigns placeholder intent and entities.
        """
        processed_emotion_data = []
        try:
            raw_emotion_dataset = load_dataset("go_emotions", split=split_name)
            print(f"  Loaded {len(raw_emotion_dataset)} entries from GoEmotions for '{split_name}'.")
            for entry in raw_emotion_dataset:
                canonical_emotion = self._map_goemotions_to_canonical(entry["labels"])
                emotion_entry_for_processing = {
                    "utt": entry["text"],
                    "lang": language,
                    "scenario": "general",
                    "intent": "no_intent", # Default for emotion-only data
                    "annotations": [],
                    "emotion_label_raw": canonical_emotion,
                    "source": 'GoEmotions' # Add source for tracking
                }
                
                processed_entry = self.process_data_entry(emotion_entry_for_processing)
                if processed_entry:
                    processed_emotion_data.append(processed_entry)
        except Exception as e:
            print(f"ERROR: Could not load or process GoEmotions data for split '{split_name}'. Exception: {e}")
        return processed_emotion_data

    def _load_hinglish_top_data(self, split_name: str) -> List[Dict[str, Any]]:
        """
        Loads and processes data from the Hinglish TOP dataset.
        Maps intents and slots from TOP format to canonical schema.
        Assumes 'train.tsv', 'validation.tsv', 'test.tsv' are directly in the script's directory.
        Assigns default "neutral" emotion as TOP doesn't have emotion labels.
        """
        processed_data = []
        
        # Determine the correct filename based on split_name
        actual_filename_split = split_name
        # The Hinglish TOP dataset uses 'validation.tsv' as the file name for the validation split
        if split_name == "validation":
            actual_filename_split = "validation" 
        # Construct the filepath directly in the script's directory (OK Driver/)
        filepath = os.path.join(script_dir, f"{actual_filename_split}.tsv") 
        
        if not os.path.exists(filepath):
            print(f"WARNING: Hinglish TOP data file not found: {filepath}. Skipping.")
            return []
        print(f"  Loading Hinglish TOP data from {filepath} for '{split_name}'.")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                # Assuming the first line is a header
                header = f.readline().strip().split('\t')
                
                # Find column indices (robust to column order changes)
                try:
                    # UPDATED COLUMN NAMES BASED ON YOUR PROVIDED HEADER
                    hinglish_utt_idx = header.index("cs_query") 
                    hinglish_parse_idx = header.index("cs_parse") 
                except ValueError:
                    print(f"ERROR: Missing expected columns ('cs_query' or 'cs_parse') in Hinglish TOP file: {filepath}. Skipping.")
                    return []
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) != len(header):
                        print(f"WARNING: Skipping malformed line in Hinglish TOP data: {line.strip()}")
                        continue
                    
                    utterance = parts[hinglish_utt_idx]
                    semantic_parse_str = parts[hinglish_parse_idx]
                    parsed_nlu = self._parse_top_semantic_parse(semantic_parse_str, utterance)
                    
                    hinglish_entry = {
                        "utt": utterance,
                        "lang": "hi-en", # Use a specific tag for Hinglish
                        "intent": parsed_nlu["intent"],
                        "annotations": parsed_nlu["annotations"],
                        "emotion_label_raw": "neutral", # Hinglish TOP doesn't provide emotion
                        "source": "Hinglish_TOP" # Add source for tracking
                    }
                    processed_entry = self.process_data_entry(hinglish_entry)
                    
                    if processed_entry:
                        processed_data.append(processed_entry)
            print(f"  Loaded {len(processed_data)} entries from Hinglish TOP for '{split_name}'.")
        except Exception as e:
            print(f"ERROR: Could not load or process Hinglish TOP data for split '{split_name}'. Exception: {e}")
        return processed_data

    def process_combined_dataset(self, split_name: str, massive_languages: List[str]) -> List[Dict[str, Any]]:
        """
        Combines data from MASSIVE (for NLU), GoEmotions (for emotion), and Hinglish TOP (for Hinglish NLU).
        """
        print(f"\nProcessing combined dataset for '{split_name}' split...")
        all_processed_data = []
        # 1. Process MASSIVE data for NLU (with default "neutral" emotion)
        for lang in massive_languages:
            all_processed_data.extend(self._load_massive_data_for_lang(split_name, lang))
        # 2. Process GoEmotions data for Emotion (with placeholder NLU info)
        all_processed_data.extend(self._load_go_emotions_data(split_name, language="en-US"))
        
        # 3. NEW: Process Hinglish TOP data for Hinglish NLU
        # The Hinglish TOP dataset uses 'train', 'validation', 'test' directly as filenames
        all_processed_data.extend(self._load_hinglish_top_data(split_name))
        print(f"Finished processing combined {split_name} data. Total entries: {len(all_processed_data)}")
        return all_processed_data

    def save_processed_data(self, data: List[Dict[str, Any]], filename: str):
        """Saves processed data to a JSONL (JSON Lines) file."""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"Processed data saved to {filepath}")

    def generate_schema_maps(self):
        """Saves the intent, entity, and emotion ID mappings to JSON files."""
        maps_dir = self.output_dir
        os.makedirs(maps_dir, exist_ok=True) 
        
        with open(os.path.join(maps_dir, "intent_to_id.json"), 'w', encoding='utf-8') as f:
            json.dump(self.intent_to_id, f, indent=4)
        with open(os.path.join(maps_dir, "id_to_intent.json"), 'w', encoding='utf-8') as f:
            json.dump(self.id_to_intent, f, indent=4)
        
        with open(os.path.join(maps_dir, "entity_to_id.json"), 'w', encoding='utf-8') as f:
            json.dump(self.entity_to_id, f, indent=4)
        with open(os.path.join(maps_dir, "id_to_entity.json"), 'w', encoding='utf-8') as f:
            json.dump(self.id_to_entity, f, indent=4)
        
        with open(os.path.join(maps_dir, "emotion_to_id.json"), 'w', encoding='utf-8') as f:
            json.dump(self.emotion_to_id, f, indent=4)
        with open(os.path.join(maps_dir, "id_to_emotion.json"), 'w', encoding='utf-8') as f:
            json.dump(self.id_to_emotion, f, indent=4)
        
        print(f"Intent, Entity, and Emotion ID mappings saved to {maps_dir}")

# --- Main execution block ---
if __name__ == "__main__":
    PREPARED_DATA_DIR = "prepared_data"
    MASSIVE_LANGUAGES = ['en-US', 'hi-IN'] 
    
    processor = DatasetProcessor(output_dir=PREPARED_DATA_DIR)
    
    train_data = processor.process_combined_dataset("train", massive_languages=MASSIVE_LANGUAGES)
    dev_data = processor.process_combined_dataset("validation", massive_languages=MASSIVE_LANGUAGES) 
    test_data = processor.process_combined_dataset("test", massive_languages=MASSIVE_LANGUAGES)
    
    lang_suffix = '_'.join(lang.replace('-', '') for lang in MASSIVE_LANGUAGES) + "_goemotions_hinglish" 
    
    processor.save_processed_data(train_data, f"train_processed_{lang_suffix}.jsonl")
    processor.save_processed_data(dev_data, f"dev_processed_{lang_suffix}.jsonl")
    processor.save_processed_data(test_data, f"test_processed_{lang_suffix}.jsonl")
    
    processor.generate_schema_maps()
    
    print("\nData preparation complete!")
    print(f"Total processed train entries: {len(train_data)}")
    print(f"Total processed dev entries: {len(dev_data)}")
    print(f"Total processed test entries: {len(test_data)}")
    
    if train_data:
        print("\nSample processed train entry (first one):")
        print(json.dumps(train_data[0], indent=2, ensure_ascii=False))

    # Find and print a sample that has a non-'O' tag to confirm the fix
    found_entity_sample = False
    for entry in train_data:
        # Check if there's any tag other than 'O'
        if any(tag != 'O' for tag in entry["entity_bio_tags"]):
            print("\nSample processed train entry (with detected entities):")
            print(json.dumps(entry, indent=2, ensure_ascii=False))
            found_entity_sample = True
            break
    if not found_entity_sample:
        print("\nWARNING: Still no entities found in processed train data. Check raw data annotations.")

    # Find a Hinglish sample to show
    found_hinglish_sample = False
    for entry in train_data:
        if entry.get("language") == "hi-en":
            print("\nSample processed train entry (Hinglish):")
            print(json.dumps(entry, indent=2, ensure_ascii=False))
            found_hinglish_sample = True
            break
    if not found_hinglish_sample:
        print("\nNo Hinglish sample found in processed train entries.")