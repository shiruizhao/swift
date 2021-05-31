import re
import collections
from string import Template
from itertools import product

import numpy as np
from joblib import Parallel, delayed
from pyparsing import (
    Word,
    alphanums,
    Suppress,
    Optional,
    CharsNotIn,
    Group,
    nums,
    ZeroOrMore,
    OneOrMore,
    cppStyleComment,
    printables,
)

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD


class BLOGWriter(object):

    """
    Base class for writing BLOG network file format
    """

    def __init__(self, model):
        """
        Initialise a BLOGWriter Object

        Parameters
        ----------
        model: BayesianModel Instance

        Examples
        ---------
        >>> from pgmpy.readwrite import BLOGWriter
        >>> from pgmpy.utils import get_example_model
        >>> asia = get_example_model('asia')
        >>> writer = BLOGWriter(asia)
        >>> writer
        <writer_BLOG.BLOGWriter at 0x7f05e5ea27b8>
        >>> writer.write_BLOG('asia.BLOG')
        """
        if not isinstance(model, BayesianModel):
            raise TypeError("model must be an instance of BayesianModel")
        self.model = model
        if not self.model.name:
            self.network_name = "unknown"
        else:
            self.network_name = self.model.name
        self.variable_states = self.get_states()
        self.property_tag = self.get_properties()
        self.variable_parents = self.get_parents()
        self.tables = self.get_cpds()

    def BLOG_templates(self):
        """
        Create template for writing in BLOG format
        """
        network_template = Template("network $name {\n}\n")
        # property tag may or may not be present in model,and since no of properties
        # can be more than one , will replace them accoriding to format otherwise null
        variable_template = Template(
            """type Type$name;\ndistinct Type$name $states;\n\n"""
        )
        property_template = Template("    property $prop ;\n")
        # $variable_ here is name of variable, used underscore for clarity
        probability_template = Template(
            """random Type$variable_ $variable_ ~ $parents\n\t$values;\n\n"""
        )
        return (
            variable_template,
            probability_template,
        )

    def __str__(self):
        """
        Returns the BLOG format as string
        """
        (
            variable_template,
            probability_template,
        ) = self.BLOG_templates()
        variables = self.model.nodes()
        
        network = ""

        for var in sorted(variables):
            no_of_states = str(len(self.variable_states[var]))
            variable_states = [var + s for s in self.variable_states[var]]
            states = ", ".join(variable_states)
            network += variable_template.substitute(
                name=var,
                no_of_states=no_of_states,
                states=states,
            )

        for var in sorted(variables):
            if not self.variable_parents[var]:
                parents = ""
                cpd = "Categorical({"+self.tables[var]+"})\n"
            else:
                if len(self.variable_parents[var]) == 1:
                    parents = "case "+", ".join(self.variable_parents[var])+" in"
                else:
                    parents = "case ["+", ".join(self.variable_parents[var])+"] in"
                cpd = "{"+self.tables[var]+"}"
            network += probability_template.substitute(
                variable_=var, parents=parents, values=cpd
            )

        network += "query "+var+";\n"

        return network

    def get_variables(self):
        """
        Add variables to BLOG

        Returns
        -------
        list: a list containing names of variable

        Example
        -------
        >>> from pgmpy.readwrite import BLOGReader, BLOGWriter
        >>> model = BLOGReader('dog-problem.BLOG').get_model()
        >>> writer = BLOGWriter(model)
        >>> writer.get_variables()
        ['bowel-problem', 'family-out', 'hear-bark', 'light-on', 'dog-out']
        """
        variables = self.model.nodes()
        return variables

    def get_states(self):
        """
        Add states to variable of BLOG

        Returns
        -------
        dict: dict of type {variable: a list of states}

        Example
        -------
        >>> from pgmpy.readwrite import BLOGReader, BLOGWriter
        >>> model = BLOGReader('dog-problem.BLOG').get_model()
        >>> writer = BLOGWriter(model)
        >>> writer.get_states()
        {'bowel-problem': ['bowel-problem_0', 'bowel-problem_1'],
         'dog-out': ['dog-out_0', 'dog-out_1'],
         'family-out': ['family-out_0', 'family-out_1'],
         'hear-bark': ['hear-bark_0', 'hear-bark_1'],
         'light-on': ['light-on_0', 'light-on_1']}
        """
        variable_states = {}
        cpds = self.model.get_cpds()
        for cpd in cpds:
            variable = cpd.variable
            variable_states[variable] = []
            for state in cpd.state_names[variable]:
                variable_states[variable].append(str(state))
        return variable_states

    def get_properties(self):
        """
        Add property to variables in BLOG

        Returns
        -------
        dict: dict of type {variable: list of properties }

        Example
        -------
        >>> from pgmpy.readwrite import BLOGReader, BLOGWriter
        >>> model = BLOGReader('dog-problem.BLOG').get_model()
        >>> writer = BLOGWriter(model)
        >>> writer.get_properties()
        {'bowel-problem': ['position = (335, 99)'],
         'dog-out': ['position = (300, 195)'],
         'family-out': ['position = (257, 99)'],
         'hear-bark': ['position = (296, 268)'],
         'light-on': ['position = (218, 195)']}
        """
        variables = self.model.nodes()
        property_tag = {}
        for variable in sorted(variables):
            properties = self.model.nodes[variable]
            properties = collections.OrderedDict(sorted(properties.items()))
            property_tag[variable] = []
            for prop, val in properties.items():
                property_tag[variable].append(str(prop) + " = " + str(val))
        return property_tag

    def get_parents(self):
        """
        Add the parents to BLOG

        Returns
        -------
        dict: dict of type {variable: a list of parents}

        Example
        -------
        >>> from pgmpy.readwrite import BLOGReader, BLOGWriter
        >>> model = BLOGReader('dog-problem.BLOG').get_model()
        >>> writer = BLOGWriter(model)
        >>> writer.get_parents()
        {'bowel-problem': [],
         'dog-out': ['bowel-problem', 'family-out'],
         'family-out': [],
         'hear-bark': ['dog-out'],
         'light-on': ['family-out']}
        """
        cpds = self.model.get_cpds()
        variable_parents = {}
        for cpd in cpds:
            variable_parents[cpd.variable] = cpd.variables[1:]
        return variable_parents

    def get_cpds(self):
        """
        Adds tables to BLOG

        Returns
        -------
        dict: dict of type {variable: array}

        Example
        -------
        >>> from pgmpy.readwrite import BLOGReader, BLOGWriter
        >>> model = BLOGReader('dog-problem.BLOG').get_model()
        >>> writer = BLOGWriter(model)
        >>> writer.get_cpds()
        {'bowel-problem': array([ 0.01,  0.99]),
         'dog-out': array([ 0.99,  0.97,  0.9 ,  0.3 ,  0.01,  0.03,  0.1 ,  0.7 ]),
         'family-out': array([ 0.15,  0.85]),
         'hear-bark': array([ 0.7 ,  0.01,  0.3 ,  0.99]),
         'light-on': array([ 0.6 ,  0.05,  0.4 ,  0.95])}
        """
        cpds = self.model.get_cpds()
        tables = {}
        for cpd in cpds:
            print(cpd)
            evidence = cpd.variables[1:]
            evidence_card = cpd.cardinality[1:]
            o_value = []
            print(evidence)
            if evidence:
                col_indexes = np.array(list(product(*[range(i) for i in evidence_card])))
                for m in range(np.shape(col_indexes)[0]):
                    col_name = []
                    var_name = []
                    cat_value = []
                    for k in range(np.size(col_indexes[m])):
                        var_name.append("{var}".format(var=evidence[k]))
                        col_name.append("{var}{state}".format(state=cpd.state_names[evidence[k]][col_indexes[m][k]], var=evidence[k]))
                    for j in range(cpd.variable_card):
                        row_name = "{var}{state}".format(
                            var=cpd.variable, state=cpd.state_names[cpd.variable][j]
                            )
                        index = np.shape(col_indexes)[0] * j + m
                        row_index = row_name +" -> " + str(cpd.values.ravel()[index])
                        cat_value.append(row_index)
                        

                    var_name = ', '.join(var_name)
                    cat_value= ', '.join(cat_value)
                    if len(col_name) == 1:
                        o_value_temp = ''.join([ col_name[0], " -> " , "Categorical({", cat_value, "})"])
                    else:
                        col_name = ', '.join(col_name)
                        o_value_temp = ''.join(["[ ", col_name, " ]"," -> " , "Categorical({", cat_value, "})"])
                    o_value.append(o_value_temp)
                tables[cpd.variable] = ',\n'.join(o_value)
            else:
                for j in range(cpd.variable_card):
                    row_name = "{var}({state})".format(
                            var=cpd.variable, state=cpd.state_names[cpd.variable][j]
                            )
                    index = j
                    print(row_name)
                    col_name="{var}{state}".format(state=cpd.state_names[cpd.variable][j], var=cpd.variable)
                    o_value_temp = "".join([col_name, " -> ", str(cpd.values.ravel()[index])])
                    o_value.append(o_value_temp)
                tables[cpd.variable] = ','.join(o_value)

        return tables

    def write_BLOG(self, filename):
        """
        Writes the BLOG data into a file

        Parameters
        ----------
        filename : Name of the file

        Example
        -------
        >>> from pgmpy.utils import get_example_model
        >>> from pgmpy.readwrite import BLOGReader, BLOGWriter
        >>> asia = get_example_model('asia')
        >>> writer = BLOGWriter(asia)
        >>> writer.write_BLOG(filename='asia.BLOG')
        """
        writer = self.__str__()
        with open(filename, "w") as fout:
            fout.write(writer)

from pgmpy.readwrite import BIFReader, BIFWriter
import os
from bif2blog import BLOGWriter
if __name__ ==  "__main__":
    file = "./survey.bif"
    if file.endswith(".bif"):
        print("convert BIF file:", file)
        model = BIFReader(file).get_model()
        writer = BLOGWriter(model)
        new_name = os.path.splitext(file)[0]+".blog"
        writer.write_BLOG(filename=new_name)

