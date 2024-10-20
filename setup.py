import setuptools

with open('requirements.txt') as file:
    requires = [line.strip() for line in file if not line.startswith('#')]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="research_summarizer",
    version="1.0.0",
    maintainer="Stanley Louis and Spencer Grant",
    maintainer_email="spencermgrant3@gmail.com",
    description="This project aims to use natural language processing and large language models "
                "to summarize scientific literature published during specified periods of time "
                "and covering specified topics.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/spencermg/research_summarizer",
    entry_points={
        'console_scripts':
            ['summarize=research_summarizer.__main__:main'],
    },
    packages=setuptools.find_packages(),
    install_requires=requires,
    python_requires='>=3.6',
)
