# -*- coding: utf-8 -*-

class Slot:
    def __init__ (self, name="" , mei = "", description = "", values = [], values_description = [], type="categorical"):
        self.type = type # categorical, verbatim
        self.name = name
        self.mei = mei
        self.description = description
        self.values = values
        self.values_description = values_description
        
        
    def len (self):
        return len(self.values) + 1 
        
    def get (self, x=None):        
        if x == None:
            if self.type == "verbatim":
                return [0., 0.]
            if self.type == "categorical":
                return [1.] + [0.]*len(self.values)
            else: 
                print("ERROR, not defined - should not happen, in Slot().get()! "+self.name)
        else: # normal values                    
            if self.type == "verbatim":
                return [0., 1.]
            if self.type == "categorical":
                val = [0.]*len(self.values)
                val[self.values.index(x)] = 1.
                return [0.] + val
            else: 
                print("ERROR, not defined! "+self.name)
    
    def add_value (self, x):
        if x not in self.values:
            self.values.append(x)

    def __repr__(self):
        str = "Slot [{}] in {}:\n".format(self.name, self.mei)
        str+= "\tDescription: {}\n".format(self.description)        
        if self.type is not "verbatim":
            str+= "\tValues: CATEGORICAL, {} values\n".format(len(self.values_description))
            for i in range (len(self.values)):
                str+= "\t\t{}\t{}\n".format(self.values[i], self.values_description[i])
        else:
            str+= "\tValues: VERBATIM\n"
        return str
        
class Template:
    # conditions is an array of (slot_name, slot_value) tuples
    # slot_groups is an array of slot_name arrays or [] for paragraph breaks
    def __init__(self, mei, version="default", conditions=[], slot_groups=[]):
        self.mei = mei
        self.version = version
        self.conditions = conditions
        self.slot_groups = slot_groups
    
    def __repr__(self):
        return "Template [{}] in MEI [{}]:\n\tConditions: {}\n\tGroups: {}\n".format(self.version, self.mei, self.conditions, self.slot_groups)
        
    def match(self, input_conditions):
        if self.conditions == []:
            return True
        if input_conditions == []:
            if self.conditions == []:
                return True
            else:
                return False
        match = True
        for sk,sv in self.conditions: # for all conditions in this template
            # find it in input conditions
            found = False
            for k,v in input_conditions: # 
                if k == sk:
                    found = True
                    if sv != v:
                        match = False
            if not found or not match:
                return False
        return True

def get_slot_from_list(slot_list, slot_name):
    for slot in slot_list:
        if slot.name == slot_name:
            return slot
    raise Exception("Slot not found: "+slot_name)
        
# returns a Template object from the list; WARNING, the list must contain only templates for a particular MEI.
def get_template_from_list(templates, input_conditions):
    default = None    
    # 1st search for default template
    for template in templates:
        if template.conditions == []:
            default = template
    if default is None:
        raise Exception("ERROR: get_template_from_list with conditions:{} has not found a default template!".format(input_conditions))
    if input_conditions == []:
        return default
    
    for template in templates:
        if template.match(input_conditions):
            return template
    raise Exception("ERROR: get_template_from_list with conditions:{} has not found a matching template!".format(input_conditions))
     