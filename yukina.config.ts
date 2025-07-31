import I18nKeys from "./src/locales/keys";
import type { Configuration } from "./src/types/config";

const YukinaConfig: Configuration = {
  title: "小小小同学",
  subTitle: "小小小同学的地盘",
  brandTitle: "小小小同学",

  description: "小小小同学的个人博客",

  site: "https://px-blogs.netlify.app",

  locale: "zh-CN", // set for website language and date format

  navigators: [
    {
      nameKey: I18nKeys.nav_bar_home,
      href: "/",
    },
    {
      nameKey: I18nKeys.nav_bar_archive,
      href: "/archive",
    },
    {
      nameKey: I18nKeys.nav_bar_about,
      href: "/about",
    },
    {
      nameKey: I18nKeys.nav_bar_github,
      href: "https://github.com/px6707",
    },
  ],

  username: "小小小同学",
  sign: "欲买桂花同载酒，\n终不似，少年游。",
  avatarUrl: "YxSWO13dIqOg2OP.thumb.1000_0.jpeg",
  socialLinks: [
    {
      icon: "line-md:github-loop",
      link: "https://github.com/px6707",
    },
    {
      icon: "mingcute:bilibili-line",
      link: "https://space.bilibili.com/393230248",
    },
    {
      icon: "mingcute:netease-music-line",
      link: "https://music.163.com/#/user/home?id=399193534",
    },
  ],
  maxSidebarCategoryChip: 6, // It is recommended to set it to a common multiple of 2 and 3
  maxSidebarTagChip: 12,
  maxFooterCategoryChip: 6,
  maxFooterTagChip: 24,

  banners: [
    "public/v2-0aea793a695f0368c8884b02cac48f3d_1440w.jpg",
    "public/v2-2b0bf97be6df11a2fa7711c30cfe571a_1440w.jpg",
    "public/v2-e81c476f5a7f2172a050e3784add56f3_1440w.jpg",
    "public/v2-86a32bcbcfe9c1341e26763d27aae982_1440w.jpg",
    "v2-892ef73b0a3ac37142139f0b5e93db34_1440w.webp",
    "v2-bd320272216ae716984467cb0d734826_1440w.jpg",
    "v2-d0c13ddc24a2abfbabdcfac8e5051bdc_1440w.png",
    "public/suolong5.webp",
  ],

  slugMode: "HASH", // 'RAW' | 'HASH'

  license: {
    name: "CC BY-NC-SA 4.0",
    url: "https://creativecommons.org/licenses/by-nc-sa/4.0/",
  },

  // WIP functions
  bannerStyle: "LOOP", // 'loop' | 'static' | 'hidden'
};

export default YukinaConfig;
