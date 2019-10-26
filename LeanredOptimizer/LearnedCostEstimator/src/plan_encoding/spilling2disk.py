from src.plan_encoding.encoding_plans import *


def normalize_label(labels, mini, maxi):
    labels_norm = (np.log(labels) - mini) / (maxi - mini)
    labels_norm = np.minimum(labels_norm, np.ones_like(labels_norm))
    labels_norm = np.maximum(labels_norm, np.zeros_like(labels_norm))
    return labels_norm


def merge_plans_level(level1, level2, isMapping=False):
    for idx, level in enumerate(level2):
        if idx >= len(level1):
            level1.append([])
        if isMapping:
            if idx < len(level1) - 1:
                base = len(level1[idx + 1])
                for i in range(len(level)):
                    if level[i][0] > 0:
                        level[i][0] += base
                    if level[i][1] > 0:
                        level[i][1] += base
        level1[idx] += level
    return level1


def make_data_job(plans, parameters):
    target_cost_batch = []
    target_card_batch = []
    operators_batch = []
    extra_infos_batch = []
    condition1s_batch = []
    condition2s_batch = []
    node_masks_batch = []
    samples_batch = []
    condition_masks_batch = []
    mapping_batch = []

    for plan in plans:
        target_cost = plan['cost']
        target_cardinality = plan['cardinality']
        target_cost_batch.append(target_cost)
        target_card_batch.append(target_cardinality)
        plan = plan['seq']
        operators, extra_infos, condition1s, condition2s, samples, condition_masks, mapping = encode_plan_job(plan, parameters)

        operators_batch = merge_plans_level(operators_batch, operators)
        extra_infos_batch = merge_plans_level(extra_infos_batch, extra_infos)
        condition1s_batch = merge_plans_level(condition1s_batch, condition1s)
        condition2s_batch = merge_plans_level(condition2s_batch, condition2s)
        samples_batch = merge_plans_level(samples_batch, samples)
        condition_masks_batch = merge_plans_level(condition_masks_batch, condition_masks)
        mapping_batch = merge_plans_level(mapping_batch, mapping, True)
    max_nodes = 0
    for o in operators_batch:
        if len(o) > max_nodes:
            max_nodes = len(o)
    print(max_nodes)
    print(len(condition2s_batch))
    operators_batch = np.array([np.pad(v, ((0, max_nodes - len(v)), (0, 0)), 'constant') for v in operators_batch])
    extra_infos_batch = np.array([np.pad(v, ((0, max_nodes - len(v)), (0, 0)), 'constant') for v in extra_infos_batch])
    condition1s_batch = np.array(
        [np.pad(v, ((0, max_nodes - len(v)), (0, 0), (0, 0)), 'constant') for v in condition1s_batch])
    condition2s_batch = np.array(
        [np.pad(v, ((0, max_nodes - len(v)), (0, 0), (0, 0)), 'constant') for v in condition2s_batch])
    samples_batch = np.array([np.pad(v, ((0, max_nodes - len(v)), (0, 0)), 'constant') for v in samples_batch])
    condition_masks_batch = np.array([np.pad(v, (0, max_nodes - len(v)), 'constant') for v in condition_masks_batch])
    mapping_batch = np.array([np.pad(v, ((0, max_nodes - len(v)), (0, 0)), 'constant') for v in mapping_batch])

    print('operators_batch: ', operators_batch.shape)

    operators_batch = np.array([operators_batch])
    extra_infos_batch = np.array([extra_infos_batch])
    condition1s_batch = np.array([condition1s_batch])
    condition2s_batch = np.array([condition2s_batch])
    samples_batch = np.array([samples_batch])
    condition_masks_batch = np.array([condition_masks_batch])
    mapping_batch = np.array([mapping_batch])

    target_cost_batch = normalize_label(target_cost_batch, parameters.cost_label_min, parameters.cost_label_max)
    target_card_batch = normalize_label(target_card_batch, parameters.card_label_min, parameters.card_label_max)

    return (
    target_cost_batch, target_card_batch, operators_batch, extra_infos_batch, condition1s_batch, condition2s_batch,
    samples_batch, condition_masks_batch, mapping_batch)


def chunks(arr, batch_size):
    return [arr[i:i+batch_size] for i in range(0, len(arr), batch_size)]

def save_data_job(plans, parameters, istest=False, batch_size=64, directory='/home/sunji/learnedcardinality/job'):
    if istest:
        suffix = 'test_'
    else:
        suffix = ''
    batch_id = 0
    for batch_id, plans_batch in enumerate(chunks(plans, batch_size)):
        print('batch_id', batch_id, len(plans_batch))
        target_cost_batch, target_cardinality_batch, operators_batch, extra_infos_batch, condition1s_batch, condition2s_batch, samples_batch, condition_masks_batch, mapping_batch = make_data_job(
            plans_batch, parameters)
        np.save(directory + '/target_cost_' + suffix + str(batch_id) + '.np', target_cost_batch)
        np.save(directory + '/target_cardinality_' + suffix + str(batch_id) + '.np', target_cardinality_batch)
        np.save(directory + '/operators_' + suffix + str(batch_id) + '.np', operators_batch)
        np.save(directory + '/extra_infos_' + suffix + str(batch_id) + '.np', extra_infos_batch)
        np.save(directory + '/condition1s_' + suffix + str(batch_id) + '.np', condition1s_batch)
        np.save(directory + '/condition2s_' + suffix + str(batch_id) + '.np', condition2s_batch)
        np.save(directory + '/samples_' + suffix + str(batch_id) + '.np', samples_batch)
        np.save(directory + '/condition_masks_' + suffix + str(batch_id) + '.np', condition_masks_batch)
        np.save(directory + '/mapping_' + suffix + str(batch_id) + '.np', mapping_batch)
        print('saved: ', str(batch_id))
