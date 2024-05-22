import json
import os


def read_case_info(department, case_name):
    pfile = f'script/{department}/{case_name}/patient.json'
    sfile = f'script/{department}/{case_name}/script.json'
    tfile = f'check_list/{department}/{case_name}.json'
    ret = {
        'token_nums': 0,
        'qa_nums': 0,
        'disease_num': 0,
        'medical_test_num': 0,
        'symptom_num': 0
    }

    with open(pfile) as fp:
        patient_info = json.load(fp)
        for value in patient_info.values():
            ret['token_nums'] += len(value)

    with open(sfile) as fp:
        script = json.load(fp)
        messages = script['messages']
        ret['qa_nums'] = len(messages) // 2

    with open(tfile) as fp:
        test_info = json.load(fp)
        ret['disease_num'] = len(test_info['diagnostic'])
        ret['medical_test_num'] = len(test_info['medical_checkup'])
        ret['symptom_num'] = len(test_info['consultation_content'])

    return ret


def statistics():
    departments = os.listdir('script')
    total = {
        'token_nums': 0,
        'qa_nums': 0,
        'disease_num': 0,
        'medical_test_num': 0,
        'symptom_num': 0
    }

    for department in departments:
        cases = os.listdir(f'script/{department}')
        department_total = {
            'token_nums': 0,
            'qa_nums': 0,
            'disease_num': 0,
            'medical_test_num': 0,
            'symptom_num': 0
        }
        for case in cases:
            result = read_case_info(department, case)
            for k, v in result.items():
                total[k] += v
                department_total[k] += v

        for k in department_total.keys():
            department_total[k] /= len(cases)

        print(department, len(cases), department_total)

    for k in total.keys():
        total[k] /= 72.0

    print(total)


if __name__ == '__main__':
    statistics()
