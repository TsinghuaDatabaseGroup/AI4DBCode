# 可以不开启验证：sql+not valid
def gen_foreign_key(original_table, original_attribute, reference_table, reference_attribute):
    sql = 'alter table {0} add constraint fk_{0}_{1} FOREIGN KEY ({2}) REFERENCES {1} ({3});'.format(original_table, reference_table, original_attribute, reference_attribute)
    return sql


def gen_foreign_key_not_valid(original_table, original_attribute, reference_table, reference_attribute):
    sql = 'alter table {0} add constraint fk_{0}_{1} FOREIGN KEY ({2}) REFERENCES {1} ({3}) not valid;'.format(original_table, reference_table, original_attribute, reference_attribute)
    return sql


def job_foreign_key():
    print(gen_foreign_key('movie_companies', 'company_type_id', 'company_type', 'id'))
    print(gen_foreign_key('movie_companies', 'company_id', 'company_name', 'id'))
    print(gen_foreign_key('movie_info_idx', 'info_type_id', 'info_type', 'id'))
    print(gen_foreign_key('movie_info', 'info_type_id', 'info_type', 'id'))
    print(gen_foreign_key('person_info', 'info_type_id', 'info_type', 'id'))
    print(gen_foreign_key('movie_keyword', 'keyword_id', 'keyword', 'id'))
    print(gen_foreign_key('title', 'kind_id', 'kind_type', 'id'))
    print(gen_foreign_key('aka_title', 'kind_id', 'kind_type', 'id'))
    print(gen_foreign_key('movie_link', 'linked_movie_id', 'title', 'id'))
    print(gen_foreign_key('movie_link', 'link_type_id', 'link_type', 'id'))
    print(gen_foreign_key('aka_title', 'movie_id', 'title', 'id'))
    print(gen_foreign_key('cast_info', 'movie_id', 'title', 'id'))
    print(gen_foreign_key('complete_cast', 'movie_id', 'title', 'id'))
    print(gen_foreign_key('movie_companies', 'movie_id', 'title', 'id'))
    print(gen_foreign_key('movie_info_idx', 'movie_id', 'title', 'id'))
    print(gen_foreign_key('movie_info', 'movie_id', 'title', 'id'))
    print(gen_foreign_key('movie_link', 'movie_id', 'title', 'id'))
    print(gen_foreign_key('movie_keyword', 'movie_id', 'title', 'id'))
    print(gen_foreign_key('aka_name', 'person_id', 'name', 'id'))
    print(gen_foreign_key('cast_info', 'person_id', 'name', 'id'))
    print(gen_foreign_key('person_info', 'person_id', 'name', 'id'))
    print(gen_foreign_key('cast_info', 'person_role_id', 'char_name', 'id'))
    print(gen_foreign_key('cast_info', 'role_id', 'role_type', 'id'))
    print(gen_foreign_key('complete_cast', 'subject_id', 'comp_cast_type', 'id'))
    print(gen_foreign_key('complete_cast', 'status_id', 'comp_cast_type', 'id'))


def xuetang_foreign_key():
    print(gen_foreign_key_not_valid('auth_userprofile', 'user_id', 'auth_user', 'id'))
    print(gen_foreign_key_not_valid('bbs_comment', 'user_id', 'auth_user', 'id'))
    print(gen_foreign_key_not_valid('courseware_studentmodule', 'student_id', 'auth_user', 'id'))
    print(gen_foreign_key_not_valid('miniprogram_discussion', 'user_id', 'auth_user', 'id'))
    print(gen_foreign_key_not_valid('miniprogram_discussion_vote', 'user_id', 'auth_user', 'id'))
    print(gen_foreign_key_not_valid('newcloud_courseregister', 'user_id', 'auth_user', 'id'))
    print(gen_foreign_key_not_valid('newcloud_courseregister', 'user_org_profile_id', 'organization_account_userorgprofile', 'id'))
    print(gen_foreign_key_not_valid('organization_account_userorgprofile', 'user_id', 'auth_user', 'id'))
    print(gen_foreign_key_not_valid('student_courseenrollment', 'user_id', 'auth_user', 'id'))
    print(gen_foreign_key_not_valid('submissions_score', 'student_item_id', 'submissions_studentitem', 'id'))
    print(gen_foreign_key_not_valid('submissions_score', 'submission_id', 'submissions_submission', 'id'))
    print(gen_foreign_key_not_valid('submissions_submission', 'student_item_id', 'submissions_studentitem', 'id'))
    print(gen_foreign_key_not_valid('submissions_scoresummary', 'student_item_id', 'submissions_studentitem', 'id'))
