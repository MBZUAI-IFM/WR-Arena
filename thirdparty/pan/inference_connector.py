import requests
import os
import json
from PIL import Image
import tempfile
import mediapy
from datetime import datetime
import time
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.change_fps import adjust_video_fps

class WM_inference:
    def __init__(self, api_endpoint, session_id=None, guidance_scale=6):
        self.api_endpoint = api_endpoint
        self.session_id = session_id
        self.guidance_scale = guidance_scale
        self.cat_video_paths = []
        self.frames_list = []
        self.text_list = []
        self.metadata = {"rounds": {}}
        self.metadata_path = None
        self.round_index = 0
        
        # Get server status on initialization
        self.get_server_status()
        
    def get_server_status(self):
        """Get and print server status from the load balancer"""
        try:
            response = requests.get(f"{self.api_endpoint}/stats/summary", timeout=5)
            response.raise_for_status()
            stats = response.json()
            
            # Add timestamp for UI display
            stats['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            print("\n=== Server Status Summary ===")
            print(f"Total Configured Endpoints: {stats.get('total_configured_endpoints', 'N/A')}")
            print(f"Live Endpoints Pinged Successfully: {stats.get('live_endpoints_pinged_successfully', 'N/A')}")
            print(f"Busy Endpoints (from ping): {stats.get('busy_endpoints_from_ping', 'N/A')}")
            print(f"Total Videos on Live Workers: {stats.get('total_videos_on_live_workers', 'N/A')}")
            print(f"Total States on Live Workers: {stats.get('total_states_on_live_workers', 'N/A')}")
            print(f"Tasks in Load Balancer Queue: {stats.get('tasks_in_lb_queue', 'N/A')}")
            print(f"Tasks Processing by Load Balancer: {stats.get('tasks_processing_by_lb', 'N/A')}")
            print("=============================\n")
            
            return stats
        except Exception as e:
            print(f"Unable to retrieve server status: {e}")
            return None
        
    def _poll_task_status(self, task_id, timeout=600):
        """Polls the task status endpoint until completion or timeout."""
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                raise Exception(f"Timeout waiting for task {task_id} to complete.")
            
            try:
                response = requests.get(f"{self.api_endpoint}/task/status/{task_id}", timeout=10)
                response.raise_for_status()
                status_data = response.json()
                
                task_status = status_data.get("status")
                if task_status == "completed":
                    return status_data.get("result")
                elif task_status == "failed":
                    error_message = status_data.get("error_message", "Task failed with no specific error message.")
                    raise Exception(f"Task {task_id} failed: {error_message}")
                elif task_status in ["queued", "pending", "processing"]:
                    # Task is still in progress, wait and poll again
                    print(f"Task {task_id} status: {task_status}. Polling again in 5 seconds...")
                    time.sleep(10) # Wait 5 seconds before next poll
                else:
                    raise Exception(f"Unknown task status '{task_status}' for task {task_id}.")
            except requests.exceptions.RequestException as e:
                # Log error and continue polling, or raise if it's a persistent issue
                print(f"Error polling task {task_id}: {e}. Retrying...")
                time.sleep(10) # Wait before retrying on network error
            except Exception as e: # Catch other exceptions from status logic
                raise Exception(f"An error occurred while polling task {task_id}: {str(e)}")

    def inference_round(self, curr_round, prompt_dict=None, image=None, session_id=None, save_dir=None,
                       fps=8, guidance_scale=6, denoising_steps=20, random_seed=0, prev_state_id=None, prev_video_id=None):
        if session_id is not None:
            self.session_id = session_id
            
        # Create save directory if it doesn't exist
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        # For the first round, we need to send the image
        if curr_round == 1:
            # prompt_dict is now just the action_prompt string
            action_prompt = prompt_dict
            image_caption = action_prompt # Use action_prompt as image_caption
            
            # Save image to a temporary file to send in the request
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                image_path = temp_file.name
                image.save(image_path)
            
            try:
                # Prepare the multipart form data
                files = {
                    'image': ('image.png', open(image_path, 'rb'), 'image/png')
                }
                data = {
                    'first_prompt': action_prompt,
                    'image_caption': image_caption,
                    'guidance_scale': guidance_scale,
                    'denoising_steps': denoising_steps,
                    'random_seed': random_seed
                }
                
                # Send POST request to the load balancer's first_round endpoint
                response = requests.post(
                    f"{self.api_endpoint}/generate/first_round",
                    files=files,
                    data=data,
                    timeout=600  # Initial request timeout, task polling has its own timeout
                )
                
                # Make sure to close and delete the temporary file
                if os.path.exists(image_path):
                    os.unlink(image_path)
                
                # Check if the request was successful
                response.raise_for_status()
                initial_result = response.json()
                task_id = initial_result.get("task_id")
                
                if not task_id:
                    raise Exception("Failed to get task_id from first round generation request.")
                
                print(f"First round task submitted. Task ID: {task_id}. Polling for completion...")
                # Poll for the final result
                result = self._poll_task_status(task_id)
                if not result:
                    raise Exception(f"Polling for task {task_id} did not return a result.")
                
                # Get state_id and video_id for return
                state_id = result.get('new_state_id')
                video_id = result.get('video_id')
                segment_path = result.get('segment_path')
                

                
                # Save the generated video locally
                if segment_path and save_dir:
                    local_path = os.path.join(save_dir, f"round-{curr_round}.mp4")
                    
                    adjust_video_fps(segment_path, local_path, fps)
                    print(f"Saved video for round {curr_round} to {local_path}")
                    
                    self.cat_video_paths.append(local_path)
                
                                # Save the metadata
                if save_dir:
                    # Create metadata file
                    self.metadata_path = os.path.join(save_dir, "meta_data.json")
                    self.metadata = {
                        "rounds": {
                            "1": {
                                "prompt": action_prompt, # Store only the action prompt
                                "state_id": state_id,
                                "video_id": video_id,
                                "video_filename": os.path.basename(local_path)
                            }
                        }
                    }
                    with open(self.metadata_path, 'w') as f:
                        json.dump(self.metadata, f, indent=2)
                
                # Return needed information including state_id and video_id
                return local_path, video_id, state_id
                
            except requests.exceptions.RequestException as e:
                raise Exception(f"Error in first round generation: {str(e)}")
                
        else:  # For subsequent rounds
            # prompt_dict is now just the action_prompt string
            action_prompt = prompt_dict
            
            if not prev_state_id or not prev_video_id:
                raise ValueError("Previous state_id and video_id are required for subsequent rounds")
            
            try:
                # Prepare the json payload
                payload = {
                    'state_id': prev_state_id,
                    'video_id': prev_video_id,
                    'new_prompt': action_prompt,
                    'guidance_scale': guidance_scale,
                    'denoising_steps': denoising_steps,
                    'random_seed': random_seed
                }
                
                # Send POST request to the load balancer's continue endpoint
                response = requests.post(
                    f"{self.api_endpoint}/generate/continue",
                    json=payload,
                    timeout=600  # Initial request timeout, task polling has its own timeout
                )
                
                # Check if the request was successful
                response.raise_for_status()
                initial_result = response.json()
                task_id = initial_result.get("task_id")

                if not task_id:
                    raise Exception("Failed to get task_id from continue generation request.")

                print(f"Continue round task submitted. Task ID: {task_id}. Polling for completion...")
                # Poll for the final result
                result = self._poll_task_status(task_id)
                if not result:
                    raise Exception(f"Polling for task {task_id} did not return a result.")
                
                # Get new state_id for the next round
                new_state_id = result.get('new_state_id')
                segment_path = result.get('segment_path')
                
                # Save the metadata

                
                # Save the generated video locally
                if segment_path and save_dir:
                    local_path = os.path.join(save_dir, f"round-{curr_round}.mp4")
                    single_path = os.path.join(save_dir, f"round-{curr_round}_single.mp4")
                    
                    # Copy the video from the server-generated path
                    adjust_video_fps(segment_path, local_path, fps)
                    print(f"Saved video for round {curr_round} to {local_path}")
                    
                    # Copy to the single path as well (required by the gradio app)
                    adjust_video_fps(segment_path, single_path, fps)
                    self.cat_video_paths.append(single_path)
                    
                    # Load the video into frames_list
                    try:
                        video_frames = mediapy.read_video(local_path)
                        self.frames_list.append(video_frames)
                    except Exception as e:
                        print(f"Warning: Could not load video frames: {e}")

                    self.metadata['rounds'][str(curr_round)] = {
                        "prompt": action_prompt, # Store only the action prompt
                        "state_id": new_state_id,
                        "video_id": prev_video_id,
                        "video_filename": os.path.basename(single_path)
                    }
                    with open(self.metadata_path, 'w') as f:
                        json.dump(self.metadata, f, indent=2)
                # Return needed information including new_state_id and video_id
                return single_path, prev_video_id, new_state_id
                
            except requests.exceptions.RequestException as e:
                raise Exception(f"Error in continue generation: {str(e)}")
        
        # Update round index
        self.round_index = curr_round
        
        # Add the prompt to the text list
        action_prompt = prompt_dict # prompt_dict is now the action_prompt string
        self.text_list.append(action_prompt)

    def delete_video_states(self, video_id):
        """Delete all states related to a specific video ID from the server"""
        if not video_id:
            print("Warning: No video ID provided for deletion")
            return False
            
        try:
            response = requests.delete(f"{self.api_endpoint}/states/video/{video_id}", timeout=60)
            result = None
            try:
                result = response.json() # Try to parse JSON regardless of status for logging
            except ValueError: # If response body is not JSON
                result = response.text # Store raw text for logging

            if response.status_code == 200:
                print(f"Successfully deleted states for video ID: {video_id}")
                if isinstance(result, dict):
                    print(f"Response: {json.dumps(result)}") 
                else:
                    print(f"Response: {result}")
                return True
            elif response.status_code == 207: # Multi-Status
                print(f"Partial success/Multi-Status deleting states for video ID {video_id} (Status 207).")
                if isinstance(result, dict):
                    print(f"Response: {json.dumps(result)}")
                else:
                    print(f"Response: {result}")
                return False # Or True, depending on desired handling of partial success
            elif response.status_code == 404: # Video ID not found by Load Balancer
                print(f"Video ID {video_id} not found by load balancer (Status 404). Nothing to delete for this ID via LB.")
                if isinstance(result, dict):
                    print(f"Response: {json.dumps(result)}")
                else:
                    print(f"Response: {result}")
                return False # No states were deleted because the video_id was unknown to LB
            else:
                # For other non-200/207/404 codes, raise an error to be caught below
                response.raise_for_status()
                # This part should ideally not be reached if raise_for_status() works
                print(f"Received unexpected status {response.status_code} for video ID {video_id}. Response: {result}")
                return False
        except requests.exceptions.HTTPError as httpe:
            print(f"HTTP error deleting video states for video ID {video_id}: {str(httpe)}")
            # result variable holds the parsed JSON or text from the try block above
            print(f"Error Response Body: {result if result else 'No response body'}")
            return False
        except requests.exceptions.RequestException as reqe: # Catches connection errors, timeouts, etc.
            print(f"Request error deleting video states for video ID {video_id}: {str(reqe)}")
            return False
        except Exception as e: # Catch-all for other errors
            print(f"Generic error deleting video states for video ID {video_id}: {str(e)}")
            return False
            
    def delete_all_video_states(self):
        """Delete all video states from all workers via the load balancer"""
        try:
            response = requests.delete(f"{self.api_endpoint}/states/all", timeout=120)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            result = response.json()
            
            print("\n=== Delete All Video States Report ===")
            print(f"  Message: {result.get('message', 'N/A')}")
            print(f"  Videos Processed for Deletion: {result.get('videos_processed_for_deletion', 'N/A')}")
            print(f"  Videos Confirmed Removed from LB Map: {result.get('videos_confirmed_removed_from_lb_map', 'N/A')}")
            print(f"  Total State Deletions/Not Found on Worker: {result.get('total_state_deletions_or_not_found_on_worker', 'N/A')}")
            print(f"  Total Failed State Deletions on Worker: {result.get('total_failed_state_deletions_on_worker', 'N/A')}")
            print(f"  Total LB Map Inconsistencies: {result.get('total_lb_map_inconsistencies', 'N/A')}")
            print(f"  Tasks Cleared from LB Queue: {result.get('tasks_cleared_from_lb_queue', 'N/A')}")
            print(f"  Task States Cleared from LB: {result.get('task_states_cleared_from_lb', 'N/A')}")
            print("====================================\n")
            
            return True, result
        except requests.exceptions.HTTPError as httpe:
            error_details = {"error": str(httpe)}
            if httpe.response is not None:
                try:
                    error_details['response_body'] = httpe.response.json()
                except ValueError:
                    error_details['response_body'] = httpe.response.text
                error_details['status_code'] = httpe.response.status_code
                print(f"HTTP error deleting all video states: {str(httpe)} - Status: {error_details['status_code']} - Response: {error_details['response_body']}")
            else:
                print(f"HTTP error deleting all video states: {str(httpe)} (No response object)")
            return False, error_details
        except requests.exceptions.RequestException as reqe: # Catches connection errors, other timeouts, etc.
            print(f"Request error deleting all video states: {str(reqe)}")
            return False, {"error": str(reqe), "type": "RequestException"}
        except Exception as e: # Catch-all for other errors like JSON parsing errors if status was OK but content isn't JSON
            print(f"Generic error deleting all video states: {str(e)}")
            return False, {"error": str(e), "type": "GenericException"}
