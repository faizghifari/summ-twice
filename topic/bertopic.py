import heapq
import numpy as np

from utils.tools import get_score

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer

class BERTopicModel:
    def __init__(self, max_len, doc_len_threshold, n_gram_range=(3, 5)):
        self.max_len = max_len
        self.doc_len_threshold = doc_len_threshold

        self.vectorizer_model = CountVectorizer(stop_words="english")
        self.ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        self.topic_model = BERTopic(nr_topics="auto", vectorizer_model=self.vectorizer_model,
                                    ctfidf_model=self.ctfidf_model, n_gram_range=n_gram_range)
        self.subtopic_model = BERTopic(nr_topics=None, vectorizer_model=self.vectorizer_model,
                                       ctfidf_model=self.ctfidf_model, n_gram_range=n_gram_range)

    def get_topic_df(self, data):
        """
        Returns the topics and their information for the given data.

        Args:
        data: A list of strings representing the input data.

        Returns:
        A dataframe containing the topics for each turn and its specific information.
        """
        topics, _ = self.topic_model.fit_transform(data)
        _ = self.topic_model.reduce_outliers(data, topics)
        topic_df = self.topic_model.get_document_info(data)
        return topic_df
    
    def get_subtopic_df(self, data):
        """
        Returns the subtopics and their information for the give topic turns.

        Args:
        data: A list of strings representing the topic.

        Returns:
        A dataframe containing the subtopics for each turn and its specific information.
        """
        self.subtopic_model.fit_transform(data)
        subtopic_df = self.subtopic_model.get_document_info(text)
        return subtopic_df
    
    def get_topic_len(self, df):
        return df.groupby('Topic')['utter_speaker'].apply(lambda x: sum(len(t.split()) for t in x)).sort_values(ascending=False)

    def iterative_recluster(self, df):
        skipped_idx = []
        topic_len = self.get_topic_len(df)
        exc_idx = topic_len[topic_len > self.max_len].index.tolist()
        
        while exc_idx and exc_idx != skipped_idx:
            for idx in exc_idx:
                if idx not in skipped_idx:
                    topic_df = df[df['Topic'] == idx]
                    text = topic_df['Document'].values.tolist()
                    subtopic_df = self.get_subtopic_df(text)
                    subtopics = subtopic_df['Topic'].unique().tolist()
                    if len(subtopics) == 1:
                        skipped_idx.append(idx)
                    else:
                        subtopic_df.index = topic_df.index
                        subtopic_df['Topic'] = subtopic_df['Topic'] + topic_len.index.max() + 2
                        df.loc[subtopic_df.index, 'Topic'] = subtopic_df['Topic']
                        topic_len = self.get_topic_len(df)
                        exc_idx = topic_len[topic_len > self.max_len].index.tolist()
                        break
        
        for topic in exc_idx:
            rows = df[df['Topic'] == topic]
            # compute scores for each row
            scores = [get_score(row['Document']) for _, row in rows.iterrows()]
            
            # remove the least important rows until topic length is below MAX_TOKENS
            while topic_len[topic] > self.max_len:
                # get index of the least important row
                min_score_idx = heapq.nsmallest(1, range(len(scores)), key=scores.__getitem__)[0]
                
                # remove row from dataframe and update topic length and scores
                row_to_remove = rows.iloc[min_score_idx]
                df.drop(row_to_remove.name, inplace=True)
                topic_len = self.get_topic_len(df)
                scores[min_score_idx] = self.max_len

        return df
    
    def remove_noise_topic(self, df):
        # iterate through each topic and compute average document length
        for topic in df['Topic'].unique():
            topic_docs = df[df['Topic'] == topic]['Document']
            
            # remove stopwords from documents and compute their length
            doc_lengths = [get_score(doc) for doc in topic_docs]
            
            # compute average document length
            avg_doc_len = sum(doc_lengths) / len(doc_lengths)
            
            # remove rows belonging to topic if average document length is below threshold
            if avg_doc_len < self.doc_len_threshold:
                df = df[df['Topic'] != topic]
        
        return df
    
    def rearrange_topic_value(self, df):
        unique_topics = df['Topic'].unique()
        topic_map = {topic: idx for idx, topic in enumerate(sorted(unique_topics))}
        df['Topic'] = df['Topic'].map(topic_map)

        return df
    
    def get_topics(self, utter, utter_speaker, remove_noise=True):
        df = self.get_topic_df(utter)
        df['utter_speaker'] = utter_speaker

        df = self.iterative_recluster(df)
        if remove_noise:
            df = self.remove_noise_topic(df)
        df = self.rearrange_topic_value(df)

        return df
