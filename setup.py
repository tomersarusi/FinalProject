from setuptools import setup, find_packages, Extension


setup(
    name="spkmeansmodule",
    version="1.0.0",
    author="T&Y",
    author_email="Duck@you.com",
    description="spkmeansmodule in c",
    install_requires=['invoke'],
    packages=find_packages(),
    license='GPL-2',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    ext_modules=[
        Extension(
            "spkmeansmodule",
            ["spkmeansmodule.c", "spkmeans.c", "kmeans.c"]
        ),
    ]
)