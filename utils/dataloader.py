import json

from utils.tools import clean_data

class DataLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_data(self) -> Tuple[List[str], List[str]]:
        """
        Loads dataset from the data_path file.

        Returns:
        dataset: A list of meetings data from the data_path.
        """
        dataset = []
        with open(self.data_path) as f:
            queries = []
            targets = []
            utter = []
            utter_speaker = []
            data = json.load(f)

            for query_type in ['general_query_list', 'specific_query_list']:
                for q in data[query_type]:
                    queries.append(q['query'])
                    targets.append(q['answer'])
            for turn in data['meeting_transcripts']:
                content = clean_data(turn['content'])
                utter.append(content)
                utter_speaker.append(turn['speaker'] + ' : ' + content)
            dataset.append({
                'queries': queries,
                'targets': targets,
                'utter': utter,
                'utter_speaker': utter_speaker,
            })
        return dataset
