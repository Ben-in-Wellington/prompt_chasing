import numpy as np
from server import PromptServer
import urllib.request
import time

class PromptChase:
    _cached_hidden_string = None
    _last_cache_update = 0
    _cache_duration = 900

    """
    A ComfyUI custom node that enables the user to play the Prompt Chase game as explained at www.prompt-chase.com. 
    
    Make sure you also have a node to display your score!
    """
    
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "positive_string": ("STRING", {
                    "multiline": True,
                    "default": "Enter your Guess here"
                })
            }
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    FUNCTION = "calculate_score"
    CATEGORY = "games"
    
    def calculate_score(self, clip, positive_string):
        current_time = time.time()
        if PromptChase._cached_hidden_string is None or (current_time - PromptChase._last_cache_update) > PromptChase._cache_duration:
            PromptChase._update_cache()
            PromptChase._last_cache_update = current_time
            
        hidden_string = PromptChase._cached_hidden_string

        tokens = clip.tokenize(positive_string)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)

        levenshtein_distance = self.levenshtein_distance(positive_string, hidden_string)
        
        return ([[cond, {"pooled_output": pooled}]], levenshtein_distance)
        
    @classmethod
    def _update_cache(cls):
        url = 'https://raw.githubusercontent.com/Ben-in-Wellington/prompt_chasing/main/daily_prompt'
        with urllib.request.urlopen(url) as response:
            html_content = response.read().decode('utf-8')

        start_marker = '<p id="content">'
        end_marker = '</p>'
        start_idx = html_content.find(start_marker) + len(start_marker)
        end_idx = html_content.find(end_marker, start_idx)
        cls._cached_hidden_string = html_content[start_idx:end_idx].strip() if start_idx < end_idx else "Default fallback string if not found."
    
    def levenshtein_distance(self, str1, str2):
        if len(str1) > len(str2):
            str1, str2 = str2, str1

        distances = range(len(str1) + 1)
        for index2, char2 in enumerate(str2):
            new_distances = [index2 + 1]
            for index1, char1 in enumerate(str1):
                if char1 == char2:
                    new_distances.append(distances[index1])
                else:
                    new_distances.append(1 + min((distances[index1], distances[index1 + 1], new_distances[-1])))
            distances = new_distances
        return distances[-1]

NODE_CLASS_MAPPINGS = {
    "PromptChase": PromptChase
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptChase": "Prompt Chase"
}
