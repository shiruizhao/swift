import re
import collections
from string import Template
from itertools import product
import os

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

import random
class SPPLWriter(object):

    """
    Base class for writing SPPL network file format
    """

    def __init__(self, model):
        """
        Initialise a SPPLWriter Object

        Parameters
        ----------
        model: BayesianModel Instance

        Examples
        ---------
        >>> from pgmpy.readwrite import SPPLWriter
        >>> from pgmpy.utils import get_example_model
        >>> asia = get_example_model('asia')
        >>> writer = SPPLWriter(asia)
        >>> writer
        <writer_SPPL.SPPLWriter at 0x7f05e5ea27b8>
        >>> writer.write_SPPL('asia.SPPL')
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
        self.tables, self.cpd_list, self.queries_id_list, self.queries_list = self.get_cpds()

    def SPPL_templates(self):
        """
        Create template for writing in SPPL format
        """
        probability_template = Template(
            """$values\n\n"""
        )
        return (
            probability_template,
        )

    def __str__(self):
        """
        Returns the SPPL format as string
        """
        (
            probability_template,
        ) = self.SPPL_templates()
        variables = self.model.nodes()
        
        network = ""

        for var in (self.cpd_list):
            cpd = self.tables[var]
            network += probability_template.substitute(
                variable_=var, values=cpd
            )


        network_queries = ','.join(self.queries_list)
        network_queries = "events = ["+network_queries+"]"

        return network, network_queries

    def get_variables(self):
        """
        Add variables to SPPL

        Returns
        -------
        list: a list containing names of variable

        Example
        -------
        >>> from pgmpy.readwrite import SPPLReader, SPPLWriter
        >>> model = SPPLReader('dog-problem.SPPL').get_model()
        >>> writer = SPPLWriter(model)
        >>> writer.get_variables()
        ['bowel-problem', 'family-out', 'hear-bark', 'light-on', 'dog-out']
        """
        variables = self.model.nodes()
        return variables

    def get_states(self):
        """
        Add states to variable of SPPL

        Returns
        -------
        dict: dict of type {variable: a list of states}

        Example
        -------
        >>> from pgmpy.readwrite import SPPLReader, SPPLWriter
        >>> model = SPPLReader('dog-problem.SPPL').get_model()
        >>> writer = SPPLWriter(model)
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
        Add property to variables in SPPL

        Returns
        -------
        dict: dict of type {variable: list of properties }

        Example
        -------
        >>> from pgmpy.readwrite import SPPLReader, SPPLWriter
        >>> model = SPPLReader('dog-problem.SPPL').get_model()
        >>> writer = SPPLWriter(model)
        >>> writer.get_properties()
        {'bowel-problem': ['position = (335, 99)'],
         'dog-out': ['position = (300, 195)'],
         'family-out': ['position = (257, 99)'],
         'hear-bark': ['position = (296, 268)'],
         'light-on': ['position = (218, 195)']}
        """
        variables = self.model.nodes()
        property_tag = {}
        for variable in (variables):
            properties = self.model.nodes[variable]
            properties = collections.OrderedDict((properties.items()))
            property_tag[variable] = []
            for prop, val in properties.items():
                property_tag[variable].append(str(prop) + " = " + str(val))
        return property_tag

    def get_parents(self):
        """
        Add the parents to SPPL

        Returns
        -------
        dict: dict of type {variable: a list of parents}

        Example
        -------
        >>> from pgmpy.readwrite import SPPLReader, SPPLWriter
        >>> model = SPPLReader('dog-problem.SPPL').get_model()
        >>> writer = SPPLWriter(model)
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
        Adds tables to SPPL

        Returns
        -------
        dict: dict of type {variable: array}

        Example
        -------
        >>> from pgmpy.readwrite import SPPLReader, SPPLWriter
        >>> model = SPPLReader('dog-problem.SPPL').get_model()
        >>> writer = SPPLWriter(model)
        >>> writer.get_cpds()
        {'bowel-problem': array([ 0.01,  0.99]),
         'dog-out': array([ 0.99,  0.97,  0.9 ,  0.3 ,  0.01,  0.03,  0.1 ,  0.7 ]),
         'family-out': array([ 0.15,  0.85]),
         'hear-bark': array([ 0.7 ,  0.01,  0.3 ,  0.99]),
         'light-on': array([ 0.6 ,  0.05,  0.4 ,  0.95])}
        """
        cpds = self.model.get_cpds()
        tables = {}
        queries_list = []

        cpd_list = []
        old_cpd_list=0
        while (len(cpd_list) != len(cpds)):
            old_cpd_list = len(cpd_list)
            for cpd in cpds:
                append_flag = True
                evidence = cpd.variables[1:]
                evidence_card = cpd.cardinality[1:]
                o_value = []
                if cpd.variable in cpd_list:
                    append_flag = False
                else:
                    if evidence:
                        print(cpd.variable)
                        for i in range(len(evidence)):
                            if evidence[i] in cpd_list:
                                append_flag = True
                                continue
                            else:
                                append_flag = False
                                break
                        if append_flag:
                            cpd_list.append(cpd.variable)

                            col_indexes = np.array(list(product(*[range(i) for i in evidence_card])))

                            iter_no = []
                            edge_num = 1
                            for k in range(np.size(col_indexes[0])):
                                iter_temp = len(cpd.state_names[evidence[np.size(col_indexes[0])-k-1]])
                                edge_num = edge_num * iter_temp
                                print(edge_num)
                                iter_no.append(edge_num)
                            print("iter_no: ")
                            print(iter_no) 

                            #for loop: eveidents
                            var_col_name_all = []
                            for m in range(np.shape(col_indexes)[0]):
                                col_name = []
                                var_name = []
                                cat_value = []
                                var_col_name = []

                                sum_value = 0

                                for j in range(cpd.variable_card):
                                    row_name = "{state}".format(
                                        state=cpd.state_names[cpd.variable][j]
                                        )
                                    index = np.shape(col_indexes)[0] * j + m

                                    if j == cpd.variable_card - 1:
                                        round_value = 1.0 - sum_value
                                        row_index = "\'"+row_name +"\' : " + str(round_value)
                                    else:
                                        row_index = "\'"+row_name +"\' : " + str(cpd.values.ravel()[index])
                                    sum_value += cpd.values.ravel()[index]
                                    cat_value.append(row_index)
                                    
                                for k in range(np.size(col_indexes[m])):
                                    var = evidence[k]
                                    state = cpd.state_names[evidence[k]][col_indexes[m][k]]
                                    var_col = "({var} == \'{state}\')".format(var=var, state=state)
                                    var_name.append("{var}".format(var=var))
                                    col_name.append("{state}".format(state=state))
                                    print('{m} {k} {var_col} {var_col_name_all}'.format(m=m, k=k, var_col=var_col, var_col_name_all=var_col_name_all))
                                    if m == 0:
                                        var_col_name.append('if ')
                                        var_col_name.append(var_col)
                                        var_col_name.append(':\n')
                                        var_col_name.append('\t'*(k+1))
                                        var_col_name_all.append(var_col)
                                    else:
                                        start_flag = False
                                        end_flag = False
                                        #del redudent item
                                        for iter_k in range(np.size(col_indexes[m]) - 1):
                                            if iter_k == 0:
                                                iter_edge = iter_no[0]
                                            else:
                                                iter_edge = int(iter_no[iter_k] / iter_no[iter_k - 1])

                                            if (k == 0) & (m % iter_no[iter_k] == 0):
                                                print("del")
                                                del var_col_name_all[-iter_edge:]


                                        #end flag
                                        reverse_k = np.size(col_indexes[m]) - k - 1
                                        if reverse_k == 0:
                                            if m % iter_no[0] == iter_no[0] - 1:
                                                print("k = reverse k end flag")
                                                end_flag = True
                                        else:
                                            if m % iter_no[reverse_k] == iter_no[reverse_k] - iter_no[reverse_k-1]:
                                                print("end flag")
                                                end_flag = True

                                        #start flag
                                        if m % iter_no[reverse_k] == 0:
                                            start_flag = True
                                            print("start flag")

                                        if any(var_col in s for s in var_col_name_all):
                                            print("mathing")
                                            var_col_name.append('\t')
                                        else:
                                            var_col_name_all.append(var_col)

                                            if start_flag:
                                                var_col_name.append('if')
                                                start_flag = False
                                                
                                                var_col_name.append("({var} == \'{state}\')".format(var=evidence[k], state=cpd.state_names[evidence[k]][col_indexes[m][k]]))
                                                var_col_name.append(':\n')
                                                var_col_name.append('\t'*(k+1))
                                            elif end_flag:
                                                var_col_name.append('else')
                                                end_flag = False
                                                
                                                var_col_name.append(':\n')
                                                var_col_name.append('\t'*(k+1))
                                            else:
                                                var_col_name.append('elif')

                                                var_col_name.append("({var} == \'{state}\')".format(var=evidence[k], state=cpd.state_names[evidence[k]][col_indexes[m][k]]))
                                                var_col_name.append(':\n')
                                                var_col_name.append('\t'*(k+1))

                                var_name = ', '.join(var_name)
                                cat_value= ', '.join(cat_value)
                                var_col_name= ''.join(var_col_name)
                                
                                o_value_temp = ''.join([var_col_name, cpd.variable, ' ~= ', "choice({", cat_value, "})\n"])
                                o_value.append(o_value_temp)
                            tables[cpd.variable] = ''.join(o_value)
                    else:
                        cpd_list.append(cpd.variable)
                        for j in range(cpd.variable_card):
                            row_name = "({state})".format(state=cpd.state_names[cpd.variable][j])
                            index = j
                            #col_name="{var}{state}".format(state=cpd.state_names[cpd.variable][j], var=cpd.variable)
                            col_name="\'{state}\'".format(state=cpd.state_names[cpd.variable][j])
                            o_value_temp = "".join([col_name, " : ", str(cpd.values.ravel()[index])])
                            o_value.append(o_value_temp)
                        o_value = ','.join(o_value)
                        o_value = cpd.variable+" ~= choice({"+o_value+"})\n"
                        tables[cpd.variable] = o_value
            
            if(len(cpd_list) > len(cpds)):
                break
                exit()

        queries_id_list = []
        for cpd in cpds:
            cpd_var = cpd.variable
            query_id = cpd_var+" = "+"Id(\'"+cpd_var+"\')"
            queries_id_list.append(query_id)


        #single marginal queries
        j = 0
        randomlist = [random.randint(0,len(cpds)-1) for x in range(50)]
        for i in range(len(randomlist)):
            index = randomlist[i]
            cpd_var = cpds[index].variable
            cpd_state_index=random.randint(0, len(cpds[index].state_names[cpd_var])-1)
            query_event = cpd_var + " << {\'" + cpds[index].state_names[cpd_var][cpd_state_index]+"\'}"
            queries_list.append(query_event)
            j+=1

        #all marginal queries
        for i in range(50):
            all_marginal_query = []
            for cpd in cpds:
                cpd_var = cpd.variable
                cpd_state_index = random.randint(0, len(cpd.state_names[cpd_var])-1)
                query_event = "("+cpd_var + " << {\'" + cpd.state_names[cpd_var][cpd_state_index]+"\'})"
                all_marginal_query.append(query_event)
            all_marginal_query = ' & '.join(all_marginal_query)
            queries_list.append(all_marginal_query)
            j+=1

        return tables, cpd_list, queries_id_list, queries_list

    def write_SPPL(self, filename):
        """
        Writes the SPPL data into a file

        Parameters
        ----------
        filename : Name of the file

        Example
        -------
        >>> from pgmpy.utils import get_example_model
        >>> from pgmpy.readwrite import SPPLReader, SPPLWriter
        >>> asia = get_example_model('asia')
        >>> writer = SPPLWriter(asia)
        >>> writer.write_SPPL(filename='asia')
        """
        writer0, writer1 = self.__str__()
        filename = os.path.splitext(filename)[0]
        
        with open("template.py", "r") as fin:
            data = fin.readlines()

        query_filename = filename+"_benchmark.py"
        with open(query_filename, 'w') as fout:
            fout.write(''.join(data))
            fout.write("data = \'\'\'\n")
            fout.write(writer0)
            fout.write('\'\'\'\n')
            fout.write("compiler = SPPL_Compiler(data)\nnamespace = compiler.execute_module()\nmodel=namespace.model\n")
            fout.write("\n".join(self.queries_id_list))
            fout.write("\n")
            fout.write(writer1)
            fout.write("\n")

            fout.write("runtime=np.zeros(100)\n")
            fout.write("for i in range(100):\n\tstart_time=time.time()\n\tquery_prob=model.prob(events[i])\n\tend_time = time.time()\n\tprint(\"--- %s seconds ---\" % (end_time - start_time))\n")
            fout.write("\n\tprint(query_prob)\n")
            fout.write("\n\truntime[i]=end_time-start_time\n")
            fout.write("print(\"single marginal time:%s\"%np.mean(runtime[0:50]))\n")
            fout.write("print(\"all marginal time:%s\"%np.mean(runtime[50:100]))\n")

from pgmpy.readwrite import BIFReader, BIFWriter
import os
from bif2SPPL import SPPLWriter
if __name__ ==  "__main__":
    file = "./alarm.bif"
    if file.endswith(".bif"):
        print("convert BIF file:", file)
        model = BIFReader(file).get_model()
        writer = SPPLWriter(model)
        new_name = os.path.splitext(file)[0]+".SPPL"
        writer.write_SPPL(filename=new_name)

