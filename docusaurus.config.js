// // @ts-check
// // Note: type annotations allow type checking and IDEs autocompletion

// const lightCodeTheme = require('prism-react-renderer').themes.github;
// const darkCodeTheme = require('prism-react-renderer').themes.dracula;

// /** @type {import('@docusaurus/types').Config} */
// const config = {
//   title: 'Physical AI & Humanoid Robotics Textbook',
//   tagline: 'A Comprehensive Guide to AI-Integrated Robotics',
//   favicon: 'img/favicon.ico',

//   // Set the production url of your site here
//   url: 'https://your-organization.github.io',
//   // Set the /<baseUrl>/ pathname under which your site is served
//   // For GitHub pages deployment, it is often '/<organization-name>/'
//   baseUrl: '/',

//   // GitHub pages deployment config.
//   // If you aren't using GitHub pages, you don't need these.
//   organizationName: 'your-organization', // Usually your GitHub org/user name.
//   projectName: 'physical-ai-textbook', // Usually your repo name.

//   onBrokenLinks: 'throw',
//   onBrokenMarkdownLinks: 'warn',

//   // Even if you don't use internalization, you can use this field to set useful
//   // metadata like html lang. For example, if your site is Chinese, you may want
//   // to replace "en" with "zh-Hans".
//   i18n: {
//     defaultLocale: 'en',
//     locales: ['en'],
//   },

//   presets: [
//     [
//       'classic',
//       /** @type {import('@docusaurus/preset-classic').Options} */
//       ({
//         docs: {
//           sidebarPath: require.resolve('./sidebars.js'),
//           // Please change this to your repo.
//           // Remove this to remove the "edit this page" links.
//           editUrl:
//             'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
//           routeBasePath: '/', // Serve the docs at the site's root
//           path: '.', // Look for docs in the root directory instead of a docs/ subdirectory
//           // Exclude problematic markdown files from node_modules
//           exclude: [
//             '**/node_modules/**',
//           ],
//         },
//         blog: false, // Disable blog functionality
//         theme: {
//           customCss: require.resolve('./src/css/custom.css'),
//         },
//       }),
//     ],
//   ],

//   themeConfig:
//     /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
//     ({
//       // Replace with your project's social card
//       image: 'img/docusaurus-social-card.jpg',
//       navbar: {
//         title: 'Physical AI & Humanoid Robotics',
//         logo: {
//           alt: 'Physical AI Textbook Logo',
//           src: 'img/logo.png',
//         },
//         items: [
//           {
//             type: 'docSidebar',
//             sidebarId: 'textbookSidebar',
//             position: 'left',
//             label: 'Textbook',
//           },
//           {
//             href: 'https://github.com/IlsaFatima',
//             label: 'GitHub',
//             position: 'right',
//           },
//         ],
//       },
//       footer: {
//         style: 'dark',
//         links: [
//           {
//             title: 'Chapters',
//             items: [
//               {
//                 label: 'Introduction',
//                 to: '/docs/chapters/ch01-introduction',
//               },
//               {
//                 label: 'Fundamentals',
//                 to: '/docs/chapters/ch02-fundamentals',
//               },
//             ],
//           },
//           {
//             title: 'Community',
//             items: [
//               {
//                 label: 'Stack Overflows',
//                 href: 'https://stackoverflow.com/questions/tagged/docusaurus',
//               },
//               {
//                 label: 'Discord',
//                 href: 'https://discord.com/channels/@me',
//               },
//               {
//                 label: 'Linkedin',
//                 href: 'https://www.linkedin.com/in/aamash-khalid-13702a384?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app',
//               },
//             ],
//           },
//           {
//             title: 'More',
//             items: [
//               {
//                 label: 'GitHub',
//                 href: 'hhttps://github.com/M-Aamash',
//               },
//             ],
//           },
//         ],
//         copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Textbook. Built with Docusaurus.`,
//       },
//       prism: {
//         theme: lightCodeTheme,
//         darkTheme: darkCodeTheme,
//         additionalLanguages: ['python', 'bash', 'json', 'yaml'],
//       },
//     }),
// };

// module.exports = config;

// @ts-check

const lightCodeTheme = require('prism-react-renderer').themes.github;
const darkCodeTheme = require('prism-react-renderer').themes.dracula;

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI & Humanoid Robotics Textbook',
  tagline: 'A Comprehensive Guide to AI-Integrated Robotics',
  favicon: 'img/favicon.ico',

  // --- IMPORTANT: GitHub Pages Deployment ---
  url: 'https://IlsaFatima.github.io',        
  baseUrl: '/physical-ai-textbook/',          
  organizationName: 'IlsaFatima1',            
  projectName: 'physical-ai-textbook',        

  onBrokenLinks: 'ignore',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),

          editUrl:
            'https://github.com/IlsaFatima1/physical-ai-textbook/tree/main/',

          routeBasePath: '/',   
          path: 'docs',        
        },

        blog: false,

        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig: ({
    image: 'img/docusaurus-social-card.jpg',

    navbar: {
      title: 'Physical AI & Humanoid Robotics',
      logo: {
        alt: 'Physical AI Textbook Logo',
        src: 'img/logo.png',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'textbookSidebar', 
          position: 'left',
          label: 'Textbook',
        },
        {
          href: 'https://github.com/IlsaFatima1',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },

    footer: {
      style: 'dark',
      links: [
        {
          title: 'Chapters',
          items: [
            {
              label: 'Introduction',
              to: '/docs/ch01-introduction',
            },
            {
              label: 'Fundamentals',
              to: '/docs/ch02-fundamentals',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'Stack Overflow',
              href: 'https://stackoverflow.com/questions/tagged/docusaurus',
            },
            {
              label: 'Discord',
              href: 'https://discord.com',
            },
            {
              label: 'LinkedIn',
              href: 'https://www.linkedin.com/in/ilsa-fatima-107381380?utm_source=share_via&utm_content=profile&utm_medium=member_android',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/IlsaFatima1',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Textbook. Built with Docusaurus.`,
    },

    prism: {
      theme: lightCodeTheme,
      darkTheme: darkCodeTheme,
      additionalLanguages: ['python', 'bash', 'json', 'yaml'],
    },
  }),
};

module.exports = config;
