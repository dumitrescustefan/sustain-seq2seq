class Slot:
    def __init__(self, name="", description = ""):
        self.type = type # categorical, verbatim
        self.name = name
        self.description = description
        self.values = ["not-present"]
        self.values_descriptions = ["Ignore me, I'm used for programming."]
        
    def len (self):
        return len(self.values)
    
    def add_value (self, value, value_description=""):
        if value not in self.values:
            self.values.append(value)
            self.values_descriptions.append(value_description)

    def __repr__(self):
        str = "Slot [{}]:\n".format(self.name)
        str+= "\tDescription: {}\n".format(self.description)        
        if self.type is not "verbatim":
            str+= "\tValues: CATEGORICAL, {} values\n".format(len(self.values_descriptions))
            for i in range (len(self.values)):
                str+= "\t\t{}\t{}\n".format(self.values[i], self.values_descriptions[i])
        else:
            str+= "\tValues: VERBATIM\n"
        return str
    
class Slots: # this is for one MEI only
    def __init__(self):
        self.slots = []
            
    def add_slot_value_pair(self, slot_name, slot_value, slot_description = ""):
        found = False
        for slot in self.slots:            
            if slot.name == slot_name:
                found = True
                break
        if not found:
            self.slots.append(Slot(slot_name, slot_description))
        for slot in self.slots:
            if slot.name == slot_name:
                slot.add_value(slot_value)

    def get_slot_object(self, slot_name):
        for slot in self.slots:
            if slot.name == slot_name:
                return slot
        raise Exception("Slot not found: "+slot_name)
            
    def __repr__(self):
        str = "Slots object contains {} slots:\n".format(len(self.slots))
        str+= "_"*60+"\n"
        for slot in self.slots:
            str+=slot.__repr__()
        return str