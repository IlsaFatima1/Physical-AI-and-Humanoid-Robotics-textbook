const lightCodeTheme = require('prism-react-renderer').themes.github;
const darkCodeTheme = require('prism-react-renderer').themes.dracula;

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'physical-ai-textbook',
  tagline: 'A Comprehensive Guide to AI-Integrated Robotics',
  favicon: 'img/favicon.ico',

  // Railway deployment
  // URL ko env variable se handle karo taake har deploy pe change na karna pade
  url: process.env.RAILWAY_STATIC_URL || 'http://localhost:3000',
  baseUrl: '/',

  organizationName: 'IlsaFatima1',
  projectName: 'Physical-AI-and-Humanoid-Robotics-textbook',

  onBrokenLinks: 'ignore',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          routeBasePath: '/',      // textbook homepage
          path: 'docs',
          editUrl:
            'https://github.com/IlsaFatima1/Physical-AI-and-Humanoid-Robotics-textbook/tree/main/',
        },
        blog: false,
        theme: {
          customCss: [
            require.resolve('./src/css/custom.css'),
            require.resolve('./src/css/chat-component.css'),
          ],
        },
      },
    ],
  ],

  themeConfig: {
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
          href: 'https://github.com/IlsaFatima1/Physical-AI-and-Humanoid-Robotics-textbook',
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
            { label: 'Introduction', to: '/ch01-introduction' },
            { label: 'Fundamentals', to: '/ch02-fundamentals' },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'LinkedIn',
              href: 'https://www.linkedin.com/in/ilsa-fatima-107381380',
            },
            {
              label: 'Discord',
              href: 'https://discord.com',
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
      copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Textbook`,
    },

    prism: {
      theme: lightCodeTheme,
      darkTheme: darkCodeTheme,
      additionalLanguages: ['python', 'bash', 'json', 'yaml'],
    },
  },

  themes: [
    // Add any themes here
  ],

  // Environment variable for API base URL
  customFields: {
    REACT_APP_API_BASE_URL: process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000',
  },
};

module.exports = config;
