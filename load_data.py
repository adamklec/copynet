import os
import sys
import argparse
from datetime import datetime
import pandas as pd
from spacy.lang.en import English
import unicodedata
from bson import ObjectId


def make_tokenized_pair_files(data_path, start=datetime(2017, 1, 1)):

    parser = English()

    if not os.path.isdir(data_path):
        os.makedirs(data_path)

    tokenized_pairs = get_text_pairs(start)

    for pair_idx, (user_text, amy_text) in enumerate(tokenized_pairs):
            if pair_idx % 100 == 0:
                print("creating file number",  pair_idx)

            user_tokens = parser(" " .join(user_text.split()))
            user_tokens = [token.orth_.lower() for token in user_tokens]
            raw_user_string = ' '.join(user_tokens)
            if 'confidentiality notice' in raw_user_string: # don't create files for user emails we failed to parse
                continue
            user_string = unicodedata.normalize('NFKD', raw_user_string).encode('ascii', 'ignore').decode()

            cleaned_amy_text = amy_text.partition("How did I do scheduling this meeting?")[0]
            cleaned_amy_text = cleaned_amy_text.partition("Amy Ingram")[0]
            cleaned_amy_text = cleaned_amy_text.partition("Andrew Ingram")[0]
            amy_tokens = parser(" " .join(cleaned_amy_text.split()))

            if len(user_tokens) > 190 or len(amy_tokens) > 190:
                continue

            amy_tokens = [token.orth_.lower() for token in amy_tokens]
            raw_amy_string = ' '.join(amy_tokens)
            amy_string = unicodedata.normalize('NFKD', raw_amy_string).encode('ascii', 'ignore').decode()

            with open(data_path + str(pair_idx), 'w') as pair_file:
                print(user_string, file=pair_file)
                print(amy_string, file=pair_file)


def get_text_pairs(start):

    first_seen = {'firstSeen': {'$gte': start}}
    statuses = ['BUSINESS', 'DISABLED', 'PERSONAL', 'PROFESSIONAL']
    status = {'status': {'$in': statuses}}
    query = {'$and': [first_seen, status]}

    projected_fields = {'_id': 1}

    people = pd.DataFrame(list(people_coll.find(query, projected_fields)))
    people_ids = people['_id'].tolist()

    valid_meetings_list = pd.DataFrame(list(meetings_coll.find({'host.0._id': {'$in': people_ids}}, {'_id': 1})))[ '_id']
    valid_meetings_list = [ObjectId(i) for i in valid_meetings_list]
    thread_df = ThreadBuilder(start, datetime.now(), valid_meeting_list=valid_meetings_list).thread_df

    print("loading text pairs...")

    text_pairs = []
    for thread_idx, message_ids in enumerate(thread_df.emails_in_thread):
        if thread_idx % 100 == 0:
            print("loading thread number %i/%i" % (thread_idx, len(thread_df.emails_in_thread)))

        emails = [production_client.emails.emails.find_one({'messageId': message_id}) for message_id in message_ids]

        labeled_texts = []
        for email in emails:
            # TODO: what are these emails without cleanedText? Can they be skipped or should the thread be discarded?
            if all(["ICS" not in intent for intent in email['intents']]) and email.get('cleanedText') and ["ONBOARD_BCC_HOST" not in email['intents']]:
                if 'broker' in email['from']:
                    cleaned_text = email['cleanedText']
                else:
                    name = email['from'].get('name', '')
                    cleaned_text = ' '.join([email['cleanedText'], name])
                labeled_texts.append((cleaned_text, email['from'], email['to']))

        for idx in range(len(labeled_texts) - 1):
            text, fr, to = labeled_texts[idx]
            from_amy = 'broker' in fr
            next_text, next_fr, next_to = labeled_texts[idx + 1]
            next_from_amy = 'broker' in next_fr
            if not from_amy and next_from_amy and fr['person'] in [t.get('person') for t in next_to]:
                text_pairs.append((text, next_text))
    return text_pairs


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Parse data loading parameters')
    argparser.add_argument('meeting_rating_path', type=str,
                        help='The path to the meeting rating project')
    argparser.add_argument('--data_path', type=str, default='./data/',
                        help='the directory where data files will be saved')

    args = argparser.parse_args()

    sys.path.insert(0, args.meeting_rating_path)
    from customer_analytics.threadBuilder import ThreadBuilder
    from customer_analytics.constants import people_coll, meetings_coll, production_client

    make_tokenized_pair_files(args.data_path)
