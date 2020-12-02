from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='TreeST',
    version='1.0',
    install_requires=requirements,
    packages=['tree_st',
              'tree_st.ccg',
              'tree_st.tagger',
              'tree_st.tagger.encoders', 'tree_st.tagger.eval', 'tree_st.tagger.oracle',
              'tree_st.util',
              'tree_st.allennlp'],
    url='https://github.com/jakpra/treeconstructive-supertagging',
    license='Apache 2.0',
    author='Jakob Prange',
    author_email='jp1724@georgetown.edu',
    description='Tree-structured constructive CCG supertagging'
)
