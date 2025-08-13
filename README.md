## 👨‍💻 Project Overview

This repository contains the official implementation of the paper:
**Xiaochong Dong, Jun Dan, Yingyun Sun, Yang Liu, Xuemin Zhang, Shengwei Mei. Quantifying Climate Change Impacts on Renewable Energy Generation: A Super-Resolution Recurrent Diffusion Model. *CSEE Journal of Power and Energy Systems*, Accpeted.**

## 🏆 **Key Innovations**

- 🌐 **Super-Resolution Recurrent Diffusion**: A novel SRDM framework that recurrently generates hourly climate data from daily inputs, ensuring temporal continuity and high-resolution fidelity.

- 🎲 **Non-Parametric Uncertainty Modeling**: Uses diffusion dynamics to capture short-term weather variability without assuming parametric forms.
- ⚙️ **Physics-Informed Power Conversion**: Combines high-resolution climate outputs with mechanistic models to quantify renewable generation in data-scarce regions.
- 🔍 **Bias Quantification in Low-Res Data**: Reveals significant errors in power estimation when using coarse climate data, underscoring the need for super-resolution.

## 📦 System Requirements

- ```
  python~=3.11.0
  pytorch~=2.1.0
  numpy~=1.24.3
  pandas~=2.0.3
  matplotlib~=3.7.2
  ```

## 📊 SRDM

Structure of SRDM：

<img src=".\Fig\SRDM.png" width="400px" />

Super-resolution results：

<img src=".\Fig\Climate.png" width="600px" />


## 📜 Citation

If you use SRDM in your research, please cite our paper: 

```
@article{SRDM,
  title={Quantifying Climate Change Impacts on Renewable Energy Generation: A Super-Resolution Recurrent Diffusion Model},
  author={Xiaochong Dong and Jun Dan and Yingyun Sun and Yang Liu and Xuemin Zhang and Shengwei Mei},
  journal={CSEE Journal of Power and Energy Systems},
  year={2025},
  doi={XXXXXXX}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 💬 Contact

For questions and collaborations, please contact: Xiaochong Dong: dream_dxc@163.com;dream_dxc@mail.tsinghua.edu.cn

## 🌐 Acknowledgements

This work was supported by:

- National Key R&D Program of China (2022YFB2403000)
- Postdoctoral Fellowship Program and China Postdoctoral Science Foundation (BX20250414)

- China Postdoctoral Science Foundation (2025M770478)
