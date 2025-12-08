import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/physical-ai-textbook/__docusaurus/debug',
    component: ComponentCreator('/physical-ai-textbook/__docusaurus/debug', '1b0'),
    exact: true
  },
  {
    path: '/physical-ai-textbook/__docusaurus/debug/config',
    component: ComponentCreator('/physical-ai-textbook/__docusaurus/debug/config', '4ef'),
    exact: true
  },
  {
    path: '/physical-ai-textbook/__docusaurus/debug/content',
    component: ComponentCreator('/physical-ai-textbook/__docusaurus/debug/content', '02c'),
    exact: true
  },
  {
    path: '/physical-ai-textbook/__docusaurus/debug/globalData',
    component: ComponentCreator('/physical-ai-textbook/__docusaurus/debug/globalData', '58f'),
    exact: true
  },
  {
    path: '/physical-ai-textbook/__docusaurus/debug/metadata',
    component: ComponentCreator('/physical-ai-textbook/__docusaurus/debug/metadata', '647'),
    exact: true
  },
  {
    path: '/physical-ai-textbook/__docusaurus/debug/registry',
    component: ComponentCreator('/physical-ai-textbook/__docusaurus/debug/registry', '125'),
    exact: true
  },
  {
    path: '/physical-ai-textbook/__docusaurus/debug/routes',
    component: ComponentCreator('/physical-ai-textbook/__docusaurus/debug/routes', 'aa1'),
    exact: true
  },
  {
    path: '/physical-ai-textbook/',
    component: ComponentCreator('/physical-ai-textbook/', '78f'),
    routes: [
      {
        path: '/physical-ai-textbook/',
        component: ComponentCreator('/physical-ai-textbook/', 'b78'),
        routes: [
          {
            path: '/physical-ai-textbook/',
            component: ComponentCreator('/physical-ai-textbook/', 'e57'),
            routes: [
              {
                path: '/physical-ai-textbook/appendices/appendix-a-installation',
                component: ComponentCreator('/physical-ai-textbook/appendices/appendix-a-installation', '696'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/physical-ai-textbook/appendices/appendix-b-setup',
                component: ComponentCreator('/physical-ai-textbook/appendices/appendix-b-setup', 'e2d'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/physical-ai-textbook/appendices/appendix-c-troubleshooting',
                component: ComponentCreator('/physical-ai-textbook/appendices/appendix-c-troubleshooting', '218'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/physical-ai-textbook/appendices/appendix-e-code-templates',
                component: ComponentCreator('/physical-ai-textbook/appendices/appendix-e-code-templates', '6db'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/physical-ai-textbook/appendices/appendix-g-resources',
                component: ComponentCreator('/physical-ai-textbook/appendices/appendix-g-resources', '24b'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/physical-ai-textbook/ch01-introduction/',
                component: ComponentCreator('/physical-ai-textbook/ch01-introduction/', '86e'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/physical-ai-textbook/ch01-introduction/exercises/chapter1-quiz',
                component: ComponentCreator('/physical-ai-textbook/ch01-introduction/exercises/chapter1-quiz', '49c'),
                exact: true
              },
              {
                path: '/physical-ai-textbook/ch02-fundamentals/',
                component: ComponentCreator('/physical-ai-textbook/ch02-fundamentals/', '223'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/physical-ai-textbook/ch02-fundamentals/exercises/chapter2-quiz',
                component: ComponentCreator('/physical-ai-textbook/ch02-fundamentals/exercises/chapter2-quiz', '099'),
                exact: true
              },
              {
                path: '/physical-ai-textbook/ch03-ros2-architecture/',
                component: ComponentCreator('/physical-ai-textbook/ch03-ros2-architecture/', '9ad'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/physical-ai-textbook/ch03-ros2-architecture/exercises/chapter3-quiz',
                component: ComponentCreator('/physical-ai-textbook/ch03-ros2-architecture/exercises/chapter3-quiz', '418'),
                exact: true
              },
              {
                path: '/physical-ai-textbook/ch04-gazebo-simulation/',
                component: ComponentCreator('/physical-ai-textbook/ch04-gazebo-simulation/', 'ae1'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/physical-ai-textbook/ch04-gazebo-simulation/exercises/chapter4-quiz',
                component: ComponentCreator('/physical-ai-textbook/ch04-gazebo-simulation/exercises/chapter4-quiz', '1fe'),
                exact: true
              },
              {
                path: '/physical-ai-textbook/ch05-isaac-platform/',
                component: ComponentCreator('/physical-ai-textbook/ch05-isaac-platform/', '292'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/physical-ai-textbook/ch05-isaac-platform/exercises/chapter5-quiz',
                component: ComponentCreator('/physical-ai-textbook/ch05-isaac-platform/exercises/chapter5-quiz', '817'),
                exact: true
              },
              {
                path: '/physical-ai-textbook/ch06-urdf-xacro/',
                component: ComponentCreator('/physical-ai-textbook/ch06-urdf-xacro/', '7a1'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/physical-ai-textbook/ch06-urdf-xacro/exercises/chapter6-quiz',
                component: ComponentCreator('/physical-ai-textbook/ch06-urdf-xacro/exercises/chapter6-quiz', 'cca'),
                exact: true
              },
              {
                path: '/physical-ai-textbook/ch07-perception-systems/',
                component: ComponentCreator('/physical-ai-textbook/ch07-perception-systems/', 'a31'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/physical-ai-textbook/ch07-perception-systems/exercises/chapter7-quiz',
                component: ComponentCreator('/physical-ai-textbook/ch07-perception-systems/exercises/chapter7-quiz', 'b0c'),
                exact: true
              },
              {
                path: '/physical-ai-textbook/ch08-navigation/',
                component: ComponentCreator('/physical-ai-textbook/ch08-navigation/', 'b79'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/physical-ai-textbook/ch08-navigation/exercises/chapter8-quiz',
                component: ComponentCreator('/physical-ai-textbook/ch08-navigation/exercises/chapter8-quiz', '2b0'),
                exact: true
              },
              {
                path: '/physical-ai-textbook/ch09-manipulation/',
                component: ComponentCreator('/physical-ai-textbook/ch09-manipulation/', 'ea7'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/physical-ai-textbook/ch09-manipulation/exercises/chapter9-quiz',
                component: ComponentCreator('/physical-ai-textbook/ch09-manipulation/exercises/chapter9-quiz', '43f'),
                exact: true
              },
              {
                path: '/physical-ai-textbook/ch10-vla-models/',
                component: ComponentCreator('/physical-ai-textbook/ch10-vla-models/', '3f1'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/physical-ai-textbook/ch11-humanoid-design/',
                component: ComponentCreator('/physical-ai-textbook/ch11-humanoid-design/', '45f'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/physical-ai-textbook/ch12-learning-adaptation/',
                component: ComponentCreator('/physical-ai-textbook/ch12-learning-adaptation/', '329'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/physical-ai-textbook/ch13-multi-robot-systems/',
                component: ComponentCreator('/physical-ai-textbook/ch13-multi-robot-systems/', '225'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/physical-ai-textbook/ch14-safety-ethics/',
                component: ComponentCreator('/physical-ai-textbook/ch14-safety-ethics/', '525'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/physical-ai-textbook/ch15-capstone-project/',
                component: ComponentCreator('/physical-ai-textbook/ch15-capstone-project/', 'a6f'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/physical-ai-textbook/ch15-capstone-project/exercises/chapter15-quiz',
                component: ComponentCreator('/physical-ai-textbook/ch15-capstone-project/exercises/chapter15-quiz', 'bed'),
                exact: true
              },
              {
                path: '/physical-ai-textbook/getting-started/intro',
                component: ComponentCreator('/physical-ai-textbook/getting-started/intro', '39c'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/physical-ai-textbook/reference/api-reference',
                component: ComponentCreator('/physical-ai-textbook/reference/api-reference', '07d'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/physical-ai-textbook/reference/dependencies',
                component: ComponentCreator('/physical-ai-textbook/reference/dependencies', '157'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/physical-ai-textbook/reference/glossary',
                component: ComponentCreator('/physical-ai-textbook/reference/glossary', 'bc1'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/physical-ai-textbook/reference/hardware-specs',
                component: ComponentCreator('/physical-ai-textbook/reference/hardware-specs', '042'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/physical-ai-textbook/reference/quiz-template',
                component: ComponentCreator('/physical-ai-textbook/reference/quiz-template', '2b5'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/physical-ai-textbook/',
                component: ComponentCreator('/physical-ai-textbook/', 'e9b'),
                exact: true
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
