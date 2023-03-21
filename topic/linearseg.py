from utils.tools import get_score

from topic.segmenter import SegmenterInterface


class LinearSegmenter(SegmenterInterface):
    def __init__(self, max_words_per_segment, doc_len_threshold):
        """
        Initializes a LinearSegmenter instance with the maximum number of words allowed in each segment.
        """
        self.max_words_per_segment = max_words_per_segment
        self.doc_len_threshold = doc_len_threshold

    def remove_noise_segments(self, segments):
        """
        Removes segments from a list of segments if the average score of each text inside the segment is below a certain threshold.
        
        Parameters:
            - segments (list): A list of segments, where each segment is a list of strings.
        
        Returns:
            A list of segments with noise segments removed.
        """
        cleaned_segments = []
        for segment in segments:
            # Compute the average score of each text in the segment
            scores = [get_score(text) for text in segment]
            avg_score = sum(scores) / len(scores)
            
            # If the average score is above the threshold, keep the segment
            if avg_score >= self.doc_len_threshold:
                cleaned_segments.append(segment)

        return cleaned_segments

    def segmentize(self, data, **kwargs):
        """
        Divides a list of text from data into segments with a maximum number of words N.
        
        Parameters:
            - data (dict): The data contain the text to be divided.
        
        Returns:
            A list of segments text.
        """
        segments = []
        current_segment = []
        current_segment_word_count = 0

        utter_speaker = data['utter_speaker']
        for text in utter_speaker:
            words = text.split()
            
            if current_segment_word_count + len(words) <= self.max_words_per_segment:
                # If adding the current text to the current segment will keep the word count under the limit, add it
                current_segment.append(text)
                current_segment_word_count += len(words)
            else:
                # If adding the current text to the current segment will exceed the word count limit, start a new segment
                segments.append(current_segment)
                current_segment = [text]
                current_segment_word_count = len(words)
        
        # Add the final segment to the list of segments
        if current_segment:
            segments.append(current_segment)
        
        remove_noise = kwargs.get('remove_noise', True)
        if remove_noise:
            segments = self.remove_noise_segments(segments)

        texts = ['\n'.join(segment) for segment in segments]

        return texts
