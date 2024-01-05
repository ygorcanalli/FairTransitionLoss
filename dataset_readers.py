from aif360.datasets import AdultDataset, GermanDataset, CompasDataset, BankDataset

def describe_dataset(train=None, val=None, test=None):
    if train is not None:
        print("#### Training Dataset shape")
        print(train.features.shape)
    if val is not None:
        print("#### Validation Dataset shape")
        print(val.features.shape)
    print("#### Test Dataset shape")
    print(test.features.shape)
    print("#### Favorable and unfavorable labels")
    print(test.favorable_label, test.unfavorable_label)
    print("#### Protected attribute names")
    print(test.protected_attribute_names)
    print("#### Privileged and unprivileged protected attribute values")
    print(test.privileged_protected_attributes,
          test.unprivileged_protected_attributes)
    print("#### Dataset feature names")
    print(test.feature_names)


def german_dataset_reader(shuffle=True):
    #label_map = {1.0: 'Good Credit', 2.0: 'Bad Credit'}
    #protected_attribute_maps = [{1.0: 'Male', 0.0: 'Female'}]
    #data = GermanDataset(protected_attribute_names=['sex'],
    #privileged_classes=[['Male']], metadata={'label_map': label_map,
    #                    'protected_attribute_maps': protected_attribute_maps})
    data = GermanDataset()
    (dataset_expanded_train,
     dataset_test) = data.split([0.8], shuffle=shuffle)

    (dataset_train,
     dataset_val) = dataset_expanded_train.split([0.8], shuffle=shuffle)
    sens_ind = 0
    sens_attr = dataset_train.protected_attribute_names[sens_ind]

    unprivileged_groups = [{sens_attr: v} for v in
                           dataset_train.unprivileged_protected_attributes[sens_ind]]
    privileged_groups = [{sens_attr: v} for v in
                         dataset_train.privileged_protected_attributes[sens_ind]]

    describe_dataset(dataset_train, dataset_val, dataset_test)

    return dataset_expanded_train, dataset_train, dataset_val, dataset_test, unprivileged_groups, privileged_groups, sens_attr

def adult_dataset_reader(shuffle=True):
    label_map = {1.0: '>50K', 0.0: '<=50K'}
    protected_attribute_maps = [{1.0: 'Male', 0.0: 'Female'}]
    data = AdultDataset(protected_attribute_names=['sex'],
                        categorical_features=['workclass', 'education', 'marital-status',
                                              'occupation', 'relationship', 'native-country', 'race'],
                      privileged_classes=[['Male']], metadata={'label_map': label_map,
                                                               'protected_attribute_maps': protected_attribute_maps})
    (dataset_expanded_train,
     dataset_test) = data.split([0.8], shuffle=shuffle)

    (dataset_train,
    dataset_val) = dataset_expanded_train.split([0.8], shuffle=shuffle)
    sens_ind = 0
    sens_attr = dataset_train.protected_attribute_names[sens_ind]

    unprivileged_groups = [{sens_attr: v} for v in
                           dataset_train.unprivileged_protected_attributes[sens_ind]]
    privileged_groups = [{sens_attr: v} for v in
                         dataset_train.privileged_protected_attributes[sens_ind]]

    describe_dataset(dataset_train, dataset_val, dataset_test)

    return dataset_expanded_train, dataset_train, dataset_val, dataset_test, unprivileged_groups, privileged_groups, sens_attr

def compas_dataset_reader(shuffle=True):

    data = CompasDataset()
    (dataset_expanded_train,
     dataset_test) = data.split([0.8], shuffle=shuffle)

    (dataset_train,
     dataset_val) = dataset_expanded_train.split([0.8], shuffle=shuffle)
    sens_ind = 1
    sens_attr = dataset_train.protected_attribute_names[sens_ind]

    unprivileged_groups = [{sens_attr: v} for v in
                           dataset_train.unprivileged_protected_attributes[sens_ind]]
    privileged_groups = [{sens_attr: v} for v in
                         dataset_train.privileged_protected_attributes[sens_ind]]

    describe_dataset(dataset_train, dataset_val, dataset_test)

    return dataset_expanded_train, dataset_train, dataset_val, dataset_test, unprivileged_groups, privileged_groups, sens_attr

def bank_dataset_reader(shuffle=True):

    data = BankDataset()
    (dataset_expanded_train,
     dataset_test) = data.split([0.8], shuffle=shuffle)

    (dataset_train,
     dataset_val) = dataset_expanded_train.split([0.8], shuffle=shuffle)
    sens_ind = 0
    sens_attr = dataset_train.protected_attribute_names[sens_ind]

    unprivileged_groups = [{sens_attr: v} for v in
                           dataset_train.unprivileged_protected_attributes[sens_ind]]
    privileged_groups = [{sens_attr: v} for v in
                         dataset_train.privileged_protected_attributes[sens_ind]]

    describe_dataset(dataset_train, dataset_val, dataset_test)

    return dataset_expanded_train, dataset_train, dataset_val, dataset_test, unprivileged_groups, privileged_groups, sens_attr