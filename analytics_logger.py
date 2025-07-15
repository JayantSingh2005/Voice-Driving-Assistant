# analytics_logger.py
import datetime
import json
import logging
import os

# Configure logging to a file
log_directory = "logs"
os.makedirs(log_directory, exist_ok=True) # Ensure logs directory exists
log_file_path = os.path.join(log_directory, "user_analytics.log")

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# You can also add a console handler for debugging
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# console_handler.setFormatter(formatter)
# logging.getLogger().addHandler(console_handler)

class AnalyticsLogger:
    """
    A class to log NLU events and user interactions for analytics purposes.
    Currently logs to a file, but can be extended to external analytics services.
    """
    def log_nlu_event(self, user_id: str, text: str, intent: dict, emotion: dict, entities: list, debug_info: dict, response_text: str):
        """
        Logs a detailed NLU processing event.

        Args:
            user_id (str): The ID of the user.
            text (str): The raw input text from the user.
            intent (dict): The detected intent (label, confidence, source).
            emotion (dict): The detected emotion (label, confidence).
            entities (list): List of extracted entities.
            debug_info (dict): Additional debug information about the processing.
            response_text (str): The generated response text.
        """
        event_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "user_id": user_id,
            "input_text": text,
            "detected_intent": intent,
            "detected_emotion": emotion,
            "extracted_entities": entities,
            "debug_info": debug_info,
            "generated_response": response_text
        }
        # Log the event as a JSON string for easy parsing later
        logging.info(json.dumps(event_data))

    # You could add more specific logging methods here, e.g.:
    # def log_navigation_start(self, user_id, destination, start_coords, dest_coords, route_summary):
    #     event = {
    #         "timestamp": datetime.datetime.now().isoformat(),
    #         "user_id": user_id,
    #         "event_type": "navigation_start",
    #         "destination": destination,
    #         "start_coords": start_coords,
    #         "destination_coords": dest_coords,
    #         "route_summary": route_summary
    #     }
    #     logging.info(json.dumps(event))

# Initialize the logger instance once
analytics_logger = AnalyticsLogger()