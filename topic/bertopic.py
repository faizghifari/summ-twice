import heapq
import numpy as np

from umap import UMAP

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer

from utils.tools import get_score
from topic.segmenter import SegmenterInterface


class BERTopicModel(SegmenterInterface):
    def __init__(self, max_len, doc_len_threshold, n_gram_range=(3, 5)):
        self.max_len = max_len
        self.doc_len_threshold = doc_len_threshold

        self.umap_model = UMAP(random_state=42)
        self.vectorizer_model = CountVectorizer(stop_words="english")
        self.ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        self.topic_model = BERTopic(
            nr_topics="auto",
            umap_model=self.umap_model,
            vectorizer_model=self.vectorizer_model,
            ctfidf_model=self.ctfidf_model,
            n_gram_range=n_gram_range)
        self.subtopic_model = BERTopic(
            nr_topics=None, 
            vectorizer_model=self.vectorizer_model,
            ctfidf_model=self.ctfidf_model, 
            n_gram_range=n_gram_range)

    def get_topic_df(self, data):
        """
        Returns the topics and their information for the given data.

        Args:
        data: A list of strings representing the input data.

        Returns:
        A dataframe containing the topics for each turn and its specific information.
        """
        topics, _ = self.topic_model.fit_transform(data)
        try:
            _ = self.topic_model.reduce_outliers(data, topics)
        except ValueError:
            pass
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
        subtopic_df = self.subtopic_model.get_document_info(data)
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
                    try:
                        subtopic_df = self.get_subtopic_df(text)
                        subtopics = subtopic_df['Topic'].unique().tolist()
                    except (TypeError, ValueError) as e:
                        skipped_idx.append(idx)
                        continue
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
            subtopics = []
            subtopic_len = 0
            subtopic_buffer = []
            
            for idx, row in rows.iterrows():
                # compute score for the row
                score = get_score(row['Document'])
                
                # if adding the row to the subtopic buffer exceeds the max_len, 
                # assign a new topic to the subtopic buffer and add it to the dataframe
                if subtopic_len + score > self.max_len:
                    new_topic = topic_len.index.max() + 2
                    df.loc[subtopic_buffer, 'Topic'] = new_topic
                    subtopics.append(new_topic)
                    subtopic_buffer = [idx]
                    subtopic_len = score
                else:
                    subtopic_buffer.append(idx)
                    subtopic_len += score
                    
            # assign a new topic to the final subtopic buffer if it exists
            if subtopic_buffer:
                new_topic = topic_len.index.max() + 2
                df.loc[subtopic_buffer, 'Topic'] = new_topic
                subtopics.append(new_topic)

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
    
    def segmentize(self, input_data, **kwargs):
        df = self.get_topic_df(input_data['utter'])
        df['utter_speaker'] = input_data['utter_speaker']

        df = self.iterative_recluster(df)
        
        remove_noise = kwargs.get('remove_noise', True)
        if remove_noise:
            df = self.remove_noise_topic(df)
        df = self.rearrange_topic_value(df)

        topics = df['Topic'].unique()  # get unique topics in the dataframe
        topic_order = []  # initialize list to store the order of topics

        # iterate over rows in the dataframe to determine topic order
        for _, row in df.iterrows():
            if row['Topic'] not in topic_order:  # if topic not yet in order list, add it
                topic_order.append(row['Topic'])

        # create list of texts joined by '\n' for each unique topic in the order determined above
        texts = []
        for topic in topic_order:
            topic_df = df[df['Topic']==topic]
            text = '\n'.join(topic_df['utter_speaker'].tolist())
            texts.append(text)
        
        return texts
