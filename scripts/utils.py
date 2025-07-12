import json
import datetime


def unroll(list_of_lists, recursive=False):
    # function that unrolls a list of [lists of [lists of [lists of ...]]] into a single list
    if len(list_of_lists) == 0:
        return list_of_lists
    is_first_element_a_list = isinstance(list_of_lists[0], (tuple, list))
    if not is_first_element_a_list:
        return list_of_lists
    new_list = [item for l in list_of_lists for item in l]
    if recursive and is_first_element_a_list:
        return unroll(new_list, recursive=recursive)
    return new_list


def generate_string_id_from_path(path):
    simplified_path = []
    for elem in path[::-1]:
        if 'childIndex' not in elem.keys():
            child_index = 'unknown'
        else:
            child_index = elem['childIndex']
        tag_name = elem['tagName']
        elem_str = '{} ({})'.format(tag_name, child_index)
        simplified_path.append(elem_str.lower())
    str_path = ' > '.join(simplified_path)
    return str_path


def get_timestamp_from_unix_time(unix_time):
    # converte unix-time para readable timestamp yyyy-mm-dd hh:mm:ss.miliseconds
    dtime = datetime.datetime.utcfromtimestamp(unix_time / 1000)
    s = dtime.strftime("%Y-%m-%d %H:%M:%S.%f")
    return s


def is_in_valid_urls(valid_urls, url):
    for v_url in valid_urls:
        if v_url in url:
            return True
    return False


def get_events_from_json(data, create_id_from_path_list=False, start_unix_time=0, valid_urls=None):
    """
    :param data: json dict
    :create_id_from_path_list: whether to use the 'path' attribute to generate ids.
                               by default we use csspath for that.
    :create_id_from_path_list: True se quiser que o id seja criado pelo attr Path
                               False se quiser que o id seja criado pelo csspathfull
    :start_unix_time: unixtime do primeiro evento. Se maior que 0 entao todo timestamp
                      vai ser recalculado como (event_timestamp - start_unix_time).
    """
    # so para salvar as urls, talvez seja util
    all_urls = []

    ignored_event_types = ['begin_recording', 'end_recording', 'select', 'submit']
    # select = select text
    # submit = submit form (already captured by keypress or click)

    # set valid urls (events from other URLs will be ignored)
    if valid_urls is None:
        valid_urls = ['perspectivesocialnetwork.com',
                      'social-network.link',
                      'love-social.firebaseapp.com']
    else:
        assert isinstance(valid_urls, list)

    selected_events = []
    for i, event in enumerate(data):
        event_type = event['evt']

        # ignore events by their type
        if event_type in ignored_event_types:
            continue

        # ignore all events that are not related to the valid urls
        if not is_in_valid_urls(valid_urls, event['evt_data']['url']):
            continue

        # corner case because google has urls like: google.com/ref=perspective.com
        if 'google.com' in event['evt_data']['url']:
            continue

        # convert tabswitch to tabchange when the URL actually changes
        if event_type == 'tabswitch' and (len(all_urls) == 0 or all_urls[-1] != event['evt_data']['url']):
            event['evt'] = 'tabchange'
            event_type = 'tabchange'

        # remove empty spaces in strings
        if 'innerText' in event['evt_data'].keys():
            event['evt_data']['innerText'] = event['evt_data']['innerText'].strip()
            # if an element does have a description, we get the "id" attribute just to have something
            if event['evt_data']['innerText'].strip() == '':
                if len(event['evt_data']['path']) == 0:
                    event['evt_data']['innerText'] = 'empty'
                else:
                    new_inner_text = event['evt_data']['path'][-1]['uniqueId']
                    if new_inner_text is not None:
                        event['evt_data']['innerText'] = new_inner_text.lower().strip()

        # use the url if it exists in the metadata of the current event
        if 'url' in event['evt_data'].keys():
            current_url = event['evt_data']['url']
        # or get the last one
        elif len(all_urls) > 0:
            current_url = all_urls[-1]
        # otherwise we are still at the beginning of recording
        else:
            current_url = 'begin_recording'

        # keep a history of all urls
        all_urls.append(current_url)
        
        # get dom path from event
        if event_type == 'tabchange':
            xpath = "none"
        # for others events, get xpath from path list
        elif 'path' in event['evt_data'].keys() and create_id_from_path_list and len(event['evt_data']['path']) > 0:
            xpath = generate_string_id_from_path(event['evt_data']['path'])
        # or from the csspath (~xpath)
        else:
            # add csspathfull to scroll-type events
            if event_type == 'scroll':
                event['evt_data']['csspathfull'] = 'none'
            # if by now an event does not have a csspathfull, it is not useful, so we ignore it
            if 'csspathfull' not in event['evt_data'].keys():
                continue
            # set id based on csspath
            xpath = event['evt_data']['csspathfull'].strip()

        # id = URL - XPATH
        event['id'] = '{} - {}'.format(current_url, xpath)
        event['url'] = current_url
        event['xpath'] = xpath

        # convert unix-timestamp to datetime with the format YYYY-MM-DD HH:MM:SS.ms
        event['unix_time'] = event['time']
        event['time'] = get_timestamp_from_unix_time(event['time'] - start_unix_time)

        # convert scroll time to s
        if event_type == 'scroll':
            event['evt_data']['scrollTime'] /= 1000

        # add keypresses to change-type events
        if event_type == 'change':
            event['key_presses'] = 0

        # only events that survived so far will be selected
        selected_events.append(event)

    # collapse inputs into a single change
    # the logic goes as follows:
    # keypress > input > keypress > input > .... > keypress > input > change
    # but sometimes a change does not happen (idk why), so we collapse the sequence of keypresses and inputs
    # by deleting the entire sequence of inputs up to the last one, which is then renamed to a "change".
    # the duration is computed via: datetime(first_keypress) - datetime(last_input)
    # note that keypress are kept intact because we need to count the number of keypress-type events later
    del_idxs = []
    tmp_idxs = []
    for i in range(len(selected_events)):
        curr_event = selected_events[i]
        next_event = selected_events[i+1] if i + 1 <= len(selected_events) - 1 else None
        if curr_event['evt'] == 'input':
            tmp_idxs.append(i)
            if next_event is None or next_event['evt'] not in ['input', 'keypress', 'change']:
                tmp_idxs.pop(-1)
                del_idxs += list(tmp_idxs)
                curr_event['evt'] = 'change'
                curr_event['key_presses'] = len(tmp_idxs) + 1
                tmp_idxs = []
            elif next_event['evt'] == 'change':
                next_event['key_presses'] = len(tmp_idxs)
                del_idxs += list(tmp_idxs)
                tmp_idxs = []
    for i in reversed(del_idxs):
        selected_events.pop(i)

    return selected_events


def group_events_by_tabchange_session(selected_events):
    """
    Group events by tabchange events. Each group is a list of events that are bounded by tabchange events.
    This will create the structure of an episode, where each episode is a list of events.
    """
    new_selected_events = []
    inner_event = []
    for event in selected_events:
        inner_event.append(event)
        if event['evt'] == 'tabchange':
            new_selected_events.append(inner_event)
            inner_event = []
    if len(inner_event) > 0:
        new_selected_events.append(inner_event)
    assert len(selected_events) == sum(map(len, new_selected_events))
    return new_selected_events


def collapse_events(grouped_events):
    """
    Collapse sequential scrolls into a single one and delete non-enter keypresses
    """
    for i, episode in enumerate(grouped_events):
        del_idxs = []
        scrolls = 0
        for j in range(len(episode)):
            curr_event = episode[j]
            next_event = episode[j+1] if j + 1 < len(episode) else None
            if curr_event['evt'] == 'scroll':
                curr_event['evt_data']['totalScrolls'] = scrolls + 1
                if next_event is not None and next_event['evt'] == 'scroll':
                    del_idxs.append(j)
                    next_event['evt_data']['scrollTime'] += curr_event['evt_data']['scrollTime']
                    scrolls += 1
                else:
                    scrolls = 0

            elif curr_event['evt'] == 'keypress' and curr_event['evt_data']['keyCode'] != 13:
                del_idxs.append(j)

        for j in reversed(del_idxs):
            grouped_events[i].pop(j)
    return grouped_events


def move_tabchange_event_to_first_subevent(grouped_selected_events):
    """
    Move the tabchange event to the first subevent of the episode.

    Make sure that the tabchange event is always the first event of the episode, and that there is only one tabchange
    event per episode.
    """
    # first try to move to the next episode
    indices_to_delete = []
    has_to_rerun = False
    for i in range(len(grouped_selected_events)-1):
        curr_evt = grouped_selected_events[i]
        next_evt = grouped_selected_events[i+1]
        if len(curr_evt) == 0:
            continue
        if curr_evt[-1]['evt'] == 'tabchange' and next_evt[0]['evt'] != 'tabchange':
            last_sub_evt = curr_evt.pop(-1)
            next_evt.insert(0, last_sub_evt)
            if len(curr_evt) == 0:
                indices_to_delete.append(i)
            elif curr_evt[-1]['evt'] == 'tabchange':
                has_to_rerun = True
    for i in reversed(indices_to_delete):
        _ = grouped_selected_events.pop(i)

    # continue until done
    if has_to_rerun:
        return move_tabchange_event_to_first_subevent(grouped_selected_events)

    # if the next episode is already a tabchange, create a new one episode with a single tabchange in it
    tabchanges_to_add = []
    for i in range(len(grouped_selected_events)-1):
        curr_evt = grouped_selected_events[i]
        if len(curr_evt) < 2:
            continue
        if curr_evt[0]['evt'] == 'tabchange' and curr_evt[-1]['evt'] == 'tabchange':
            last_sub_evt = curr_evt.pop(-1)
            tabchanges_to_add.append((i, last_sub_evt))
    for i, evt in reversed(tabchanges_to_add):
        grouped_selected_events.insert(i+1, [evt])
    
    # continue until done
    if len(tabchanges_to_add) > 0:
        return move_tabchange_event_to_first_subevent(grouped_selected_events)

    return grouped_selected_events


def get_action_by_timestamp(grouped_selected_events, timestamp):
    # timestamp should be a str with the format:
    # hh:mm:ss or hh::ms:ss.ms
    # threshold: a str with format ss.ms

    # add ms if not present in timestamp
    if isinstance(timestamp, str):
        timestamp += '.000' if '.' not in timestamp else ''
        timestamp_obj = datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')
    else:
        assert isinstance(timestamp, datetime.datetime)
        timestamp_obj = timestamp

    best_so_far = float('inf')
    best_i = 0
    best_j = 0
    breaked = False
    for i, events in enumerate(grouped_selected_events):
        for j, sub_event in enumerate(events):
            if sub_event['evt'] in ['tabchange', 'keypress', 'scroll']:  # these are ignored for action-level smells
                continue
            sub_event_time_obj = datetime.datetime.strptime(sub_event['time'], '%Y-%m-%d %H:%M:%S.%f')
            delta = abs(sub_event_time_obj - timestamp_obj)
            t = delta.total_seconds()
            if t <= best_so_far:
                best_so_far = t
                best_i = i
                best_j = j
            else:
                breaked = True
                break
        if breaked:
            break
    return grouped_selected_events[best_i][best_j], best_i, best_j


def get_nearest_event_by_timestamp(grouped_selected_events, timestamp):
    if isinstance(timestamp, str):
        timestamp += '.000' if '.' not in timestamp else ''
        timestamp_obj = datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')
    else:
        assert isinstance(timestamp, datetime.datetime)
        timestamp_obj = timestamp
    for i, events in enumerate(grouped_selected_events):
        for j, sub_event in enumerate(events):
            if sub_event['evt'] == 'tabchange':
                continue
            sub_event_time_obj = datetime.datetime.strptime(sub_event['time'], '%Y-%m-%d %H:%M:%S.%f')
            delta = sub_event_time_obj - timestamp_obj
            t = delta.total_seconds()
            if t > 0:
                return i, j
    return len(grouped_selected_events)-1, len(grouped_selected_events[-1])-1


def get_event_by_timestamp(grouped_selected_events, timestamp, threshold='00.000'):
    # timestamp should be a str with the format:
    # hh:mm:ss or hh::ms:ss.ms
    # threshold: a str with format ss.ms

    # add ms if not present in timestamp
    if isinstance(timestamp, str):
        timestamp += '.000' if '.' not in timestamp else ''
        timestamp_obj = datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')
    else:
        assert isinstance(timestamp, datetime.datetime)
        timestamp_obj = timestamp
    threshold_obj = datetime.timedelta(seconds=int(threshold.split('.')[0]),
                                       milliseconds=int(threshold.split('.')[1]))

    best_so_far = float('inf')
    best_i = -1
    for i, events in enumerate(grouped_selected_events):
        sub_event = events[0]  # only look at the first subevent since it should be a 'tabchange'
        sub_event_time_obj = datetime.datetime.strptime(sub_event['time'], '%Y-%m-%d %H:%M:%S.%f')
        delta = (sub_event_time_obj + threshold_obj) - timestamp_obj
        t = delta.total_seconds()
        if t < 0:
            continue
        if t < best_so_far:
            best_so_far = t
            best_i = i
        else:
            break
    return grouped_selected_events[best_i], best_i


def get_events_by_timestamp(grouped_selected_events, timestamp_curr, timestamp_next,
                            threshold_curr='00.000', threshold_next='00.000', verbose=False):

    # first get the episode index for the first timestamp
    _, event_idx_curr = get_event_by_timestamp(grouped_selected_events, timestamp_curr)

    # if timestamp_next is None, we get all episodes until the rest of the list
    if timestamp_next is None:
        return grouped_selected_events[event_idx_curr:]

    # otherwise, we also get the episode index for the next timestamp
    _, event_idx_next = get_event_by_timestamp(grouped_selected_events, timestamp_next)

    # the selected events is just an slice between the episodes of the first and next timestamps
    sliced_events = grouped_selected_events[event_idx_curr:event_idx_next]

    # if the slice is not empty, we return then
    if len(sliced_events) > 0:
        return sliced_events
    else:
        # otherwise...
        # we have a tricky corner case that happens when a task is not enclosed by tabchanges
        # to circumvent retuning an empty list, we create an artificial tabchange at timestamp_curr

        # get the nearest possible event from timestamp_curr
        # i = episode id
        # j = event id
        i, j = get_nearest_event_by_timestamp(grouped_selected_events, timestamp_curr)
        event = grouped_selected_events[i][j]

        # transform the timestamp to datetime and then back to string (so formats are ok)
        timestamp_obj = datetime.datetime.strptime(timestamp_curr, '%Y-%m-%d %H:%M:%S')
        timestamp_curr_ = timestamp_obj.strftime("%Y-%m-%d %H:%M:%S.%f")

        # pretty-printing
        ts = lambda x: x.split()[1]
        if verbose:
            print('--- The interval from {} to {} is empty.'.format(ts(timestamp_curr), ts(timestamp_next)))
            print('--- First tabchange after {} is at: {}'.format(ts(timestamp_curr),
                                                                  ts(grouped_selected_events[event_idx_curr][0]['time'])))
            print('--- Creating an artificial tabchange by slicing out the episode from {} to {} at {}'.format(
                ts(grouped_selected_events[i][0]['time']),
                ts(grouped_selected_events[i][-1]['time']),
                ts(timestamp_curr_)
            ))
            print('episode {} - task {} '.format(i, j))

        # create the artificial tabchange with duration 0
        artificial_tabchange = {
            'evt': 'tabchange',
            'evt_data': {
                'url': event['evt_data']['url']
            },
            'time': timestamp_curr_,
            'unix_time': event['unix_time'],
            'id': '{} - none'.format(event['evt_data']['url']),
            'url': event['evt_data']['url'],
            'xpath': 'none',
            'labels': [],
        }

        # concat the artificial event with the events starting from j at episode i
        rest_episode = [artificial_tabchange] + grouped_selected_events[i][j:]

        # now delete these events from the i-th episode
        del grouped_selected_events[i][j:]

        # add the "new" events as a new episode at position i+1 (after i)
        grouped_selected_events.insert(i+1, rest_episode)

        # return the slice from i+1 to i+2 (i.e., a single episode)
        return grouped_selected_events[i+1:i+2]


def annotate_labels_in_selected_events(df, all_events, smells, interval_option='start', verbose=False):
    task_level_smells = {'tarefa_trabalhosa', 'tarefa_ciclica', 'muitas_camadas', 'feedback_tarefa_ausente',
                         'alta_distancia_interacao', 'repeticao_textos', 'validacao_tardia', 'tarefa_concluida'}
    action_level_smells = {'elemento_nao_descritivo', 'feedback_acao_ausente', 'acao_desnecessaria', 'acao_enganosa'}
    df = df.fillna('NaN')

    # initialize
    for episode in all_events:
        for event in episode:
            event['labels'] = []
            event['task_id'] = 0

    task_events = []
    task_smells = []
    for i in range(len(df)):
        task_curr = df.iloc[i]

        if interval_option == 'start':
            # option 1: tempo_gold represents the "start time" of the task
            timestamp_curr = df.iloc[i]['tempo_gold']

            # get the first non-empty timestamp
            timestamp_next = None
            for j in range(i+1, len(df)):
                timestamp_next = df.iloc[j]['tempo_gold']
                if timestamp_next != 'NaN':
                    break

        else:
            # option 2: tempo_gold represents the "end time" of the task
            timestamp_curr = df.iloc[i-1]['tempo_gold'] if i >= 1 else '00:00:00'

            # get the first non-empty timestamp
            timestamp_next = None
            for j in range(i, len(df) - 1):
                timestamp_next = df.iloc[j]['tempo_gold']
                if timestamp_next != 'NaN':
                    break

        # skip this task if a timestamp was not provided
        if timestamp_curr == 'NaN':
            if verbose:
                print('Task {:2d}: no timestamp. Skipping...'.format(i+1))
            continue

        if verbose:
            print('Task {:2d}: from {} to {}'.format(i+1,
                                                     timestamp_curr.split()[1],
                                                     timestamp_next.split()[1] if timestamp_next is not None else '99:99:99'))

        # get the slice of episodes
        task_events = get_events_by_timestamp(
            all_events,
            timestamp_curr=timestamp_curr,
            timestamp_next=timestamp_next,
            threshold_curr='00.000',
            threshold_next='00.000',
            verbose=verbose
        )

        # set the task_id for each episode
        for episode in task_events:
            for event in episode:
                event['task_id'] = int(i) + 1

        # get annotated smells for this task
        task_smells = [s for t, s in zip(task_curr[smells].tolist(), smells) if t != 'NaN' and str(t).strip() != '']
        for smell in task_smells:
            # if it is task-level smell, we only annotate the first tabchange of each event in the
            # selected episodes
            if smell in task_level_smells:
                for episode in task_events:
                    first_event = episode[0]  # this should be tabchange
                    first_event['labels'].append(smell)

            # otherwise, if it is an action-level smell, we annotate all events inside the selected episodes
            elif smell in action_level_smells:
                # we might have more than one annotation for action-level smells (multiple timestamps)
                timestamps = str(task_curr[smell]).split('\n')
                for timestamp in timestamps:
                    action, _, _ = get_action_by_timestamp(task_events, timestamp)
                    action['labels'].append(smell)

    return task_events, task_smells
