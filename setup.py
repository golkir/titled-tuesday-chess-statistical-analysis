from setuptools import setup, find_packages

setup(
    name='titled-tuesday-statistical-analysis',
    packages=find_packages(exclude=[]),
    include_package_data=True,
    version='1.0',
    license='Apache 2.0',
    description='Statistical analysis of Chess.com Titled Tuesday events',
    author='Kirill Goltsman',
    author_email='goltsmank@gmail.com',
    long_description_content_type='text/markdown',
    url='',
    keywords=[
        'chess games',
        'titled tuesday',
        'chess data science',
        'game accuracy'
    ],
    install_requires=[
        'beautifulsoup4==4.12.2',
        'pandas==2.0.3',
        'pymongo==4.6.1',
        'urllib3==2.1.0',
        'requests==2.31.0',
        'matplotlib==3.8.2',
        'numpy==1.22.1',
        'scipy==1.11.4',
        'seaborn==0.13.0',
        'scikit-learn==1.3.2'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Data Science',
        'License :: Apache 2.0',
        'Programming Language :: Python :: 3.6',
    ],
)
