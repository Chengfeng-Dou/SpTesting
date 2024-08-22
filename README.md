# SpTesting
Chinese data sets for standardized patient testing

Unzip the script.zip and check_list.zip files to access the dataset. Then, input your OpenAI Key into the designated field in the config file to initiate testing.

The testing entry point is located in the simulator.py file. You are required to alter the contents of the doctor.py file to load your local model for testing.

After the simulated dialogue history has been generated, proceed to evaluate the model's performance against the contents of the check_list.zip file. Please note that we have conducted tests using GPT-4 to evaluate the dialogues based on the checklist. However, we found that GPT-4's capabilities were somewhat limited. Therefore, until a more advanced version of GPT-4 is available, we highly recommend manual evaluation.

# Citation
If you use our dataset, please cite it:

```text
@article{dou2024integrating,
@inproceedings{dou-etal-2024-integrating,
    title = "Integrating Physician Diagnostic Logic into Large Language Models: Preference Learning from Process Feedback",
    author = "Dou, Chengfeng  and
      Zhang, Ying  and
      Jin, Zhi  and
      Jiao, Wenpin  and
      Zhao, Haiyan  and
      Zhao, Yongqiang  and
      Tao, Zhengwei",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.144",
    pages = "2453--2473"
}
```

# License
<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/3.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/3.0/88x31.png" /></a>
