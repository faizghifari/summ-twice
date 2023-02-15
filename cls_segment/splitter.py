
class ClusterSplitter(object):
    def __init__(self, max_model_len):
        self.max_model_len = max_model_len

    def split_utter_speaker(self, df, topic):
        topic_df = df[df['topic'] == topic].reset_index(drop=True)
        if topic_df.utter_speaker_len.sum() <= self.max_model_len:
            return ["".join(topic_df['utter_speaker'])]
        else:
            topic_df = topic_df[topic_df['utter_speaker_len'] >= 5].reset_index(drop=True)
            if topic_df.utter_speaker_len.sum() <= self.max_model_len:
                return ["".join(topic_df['utter_speaker'])]
            else:
                utter_speaker_splits = []
                current_len = 0
                segment = []
                for _, row in topic_df.iterrows():
                    if current_len + row['utter_speaker_len'] >= self.max_model_len:
                        utter_speaker_splits.append("".join(segment))
                        current_len = 0
                        segment = []
                    segment.append(row['utter_speaker'])
                    current_len += row['utter_speaker_len']
                if segment:
                    utter_speaker_splits.append("".join(segment))
            return utter_speaker_splits
