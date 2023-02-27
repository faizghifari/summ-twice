import pandas as pd

import utils.tools

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

from cls_segment.segmenter import TargetMatchSegmenter
from cls_segment.splitter import ClusterSplitter

class SegmentClusterProcessor(object):
    def __init__(self, max_model_len, max_target_sent, utter_len_threshold, n_cluster_query):
        self.max_model_len = max_model_len
        self.utter_len_threshold = utter_len_threshold
        self.n_cluster_query = n_cluster_query

        self.splitter = ClusterSplitter(max_model_len)
        self.segmenter = TargetMatchSegmenter(max_target_sent)

        self.vectorizer_model = CountVectorizer(stop_words="english")
        self.ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    # def process_cluster_summary_utter(self, utter, utter_speaker, target, query):
    def cluster_utter(self, utter, utter_speaker):
        input_len = 0
        for u in utter:
            input_len += len(u.split())
        
        if input_len <= self.max_model_len:
            return []
        
        total_clusters = (input_len // self.max_model_len) + 1
        cluster_model = KMeans(n_clusters=total_clusters)
        topic_model = BERTopic(
            nr_topics="auto", 
            vectorizer_model=self.vectorizer_model, 
            ctfidf_model=self.ctfidf_model, 
            n_gram_range=(3,5), 
            hdbscan_model=cluster_model)
        topic_model.fit_transform(utter)

        utter_len = [len(u.split()) for u in utter]
        utter_speaker_len = [len(u.split()) for u in utter_speaker]
        df = pd.DataFrame({
            "utter_speaker": utter_speaker,
            "utter": utter,
            "utter_speaker_len": utter_speaker_len,
            "utter_len": utter_len, 
            "topic": topic_model.topics_
        })

        # topic_model.find_topics(query, top_n=N_CLUSTER_QUERY)

        df2 = df[df.groupby('topic')['utter_len'].transform('mean') > self.utter_len_threshold]
        topics = df2.topic.unique()
        topics.sort()

        total_segments = 0
        cluster_data = []
        for t in topics:
            # segments_data = []
            segments = self.splitter.split_utter_speaker(df2, t)
            cluster_data.append(segments)
            total_segments += len(segments)
            # for s in segments:
            #     _, seg_summaries = self.segmenter.seg_based_on_rouge(s, target)
            #     src.append(s)
            #     tgt.append(seg_summaries)
                # cluster_data.append({
                #     "src": s,
                #     "tgt": seg_summaries,
                # })
                # segments_data.append((s, seg_summaries))
            # cluster_data.append(segments_data)

        return cluster_data, total_segments
