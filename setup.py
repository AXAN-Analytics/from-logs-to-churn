from setuptools import setup, find_packages
setup(
    name="etl-sample",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas==2.2.2","numpy==1.26.4","faker==24.9.0","duckdb","pyarrow","dash"
    ],
    extras_require={"cloud": ["s3fs"]},
    entry_points={"console_scripts": ["etl-gen=etl.cli:main"]},
)