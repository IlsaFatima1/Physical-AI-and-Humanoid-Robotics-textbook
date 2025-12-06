import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/__docusaurus/debug',
    component: ComponentCreator('/__docusaurus/debug', '5ff'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/config',
    component: ComponentCreator('/__docusaurus/debug/config', '5ba'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/content',
    component: ComponentCreator('/__docusaurus/debug/content', 'a2b'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/globalData',
    component: ComponentCreator('/__docusaurus/debug/globalData', 'c3c'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/metadata',
    component: ComponentCreator('/__docusaurus/debug/metadata', '156'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/registry',
    component: ComponentCreator('/__docusaurus/debug/registry', '88c'),
    exact: true
  },
  {
    path: '/__docusaurus/debug/routes',
    component: ComponentCreator('/__docusaurus/debug/routes', '000'),
    exact: true
  },
  {
    path: '/',
    component: ComponentCreator('/', 'be3'),
    routes: [
      {
        path: '/',
        component: ComponentCreator('/', 'd3d'),
        routes: [
          {
            path: '/',
            component: ComponentCreator('/', '2a9'),
            routes: [
              {
                path: '/appendices/appendix-a-installation',
                component: ComponentCreator('/appendices/appendix-a-installation', '028'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/appendices/appendix-b-setup',
                component: ComponentCreator('/appendices/appendix-b-setup', 'd0d'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/appendices/appendix-c-troubleshooting',
                component: ComponentCreator('/appendices/appendix-c-troubleshooting', '8c4'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/appendices/appendix-e-code-templates',
                component: ComponentCreator('/appendices/appendix-e-code-templates', '5ea'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/appendices/appendix-g-resources',
                component: ComponentCreator('/appendices/appendix-g-resources', '494'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/ch01-introduction/',
                component: ComponentCreator('/ch01-introduction/', 'd1e'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/ch01-introduction/exercises/chapter1-quiz',
                component: ComponentCreator('/ch01-introduction/exercises/chapter1-quiz', 'cdc'),
                exact: true
              },
              {
                path: '/ch02-fundamentals/',
                component: ComponentCreator('/ch02-fundamentals/', '433'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/ch02-fundamentals/exercises/chapter2-quiz',
                component: ComponentCreator('/ch02-fundamentals/exercises/chapter2-quiz', '00b'),
                exact: true
              },
              {
                path: '/ch03-ros2-architecture/',
                component: ComponentCreator('/ch03-ros2-architecture/', 'd55'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/ch03-ros2-architecture/exercises/chapter3-quiz',
                component: ComponentCreator('/ch03-ros2-architecture/exercises/chapter3-quiz', 'c9e'),
                exact: true
              },
              {
                path: '/ch04-gazebo-simulation/',
                component: ComponentCreator('/ch04-gazebo-simulation/', '943'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/ch04-gazebo-simulation/exercises/chapter4-quiz',
                component: ComponentCreator('/ch04-gazebo-simulation/exercises/chapter4-quiz', '389'),
                exact: true
              },
              {
                path: '/ch05-isaac-platform/',
                component: ComponentCreator('/ch05-isaac-platform/', '457'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/ch05-isaac-platform/exercises/chapter5-quiz',
                component: ComponentCreator('/ch05-isaac-platform/exercises/chapter5-quiz', '7ff'),
                exact: true
              },
              {
                path: '/ch06-urdf-xacro/',
                component: ComponentCreator('/ch06-urdf-xacro/', 'cd8'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/ch06-urdf-xacro/exercises/chapter6-quiz',
                component: ComponentCreator('/ch06-urdf-xacro/exercises/chapter6-quiz', '525'),
                exact: true
              },
              {
                path: '/ch07-perception-systems/',
                component: ComponentCreator('/ch07-perception-systems/', '1ac'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/ch07-perception-systems/exercises/chapter7-quiz',
                component: ComponentCreator('/ch07-perception-systems/exercises/chapter7-quiz', 'e45'),
                exact: true
              },
              {
                path: '/ch08-navigation/',
                component: ComponentCreator('/ch08-navigation/', '51a'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/ch08-navigation/exercises/chapter8-quiz',
                component: ComponentCreator('/ch08-navigation/exercises/chapter8-quiz', '684'),
                exact: true
              },
              {
                path: '/ch09-manipulation/',
                component: ComponentCreator('/ch09-manipulation/', '25c'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/ch09-manipulation/exercises/chapter9-quiz',
                component: ComponentCreator('/ch09-manipulation/exercises/chapter9-quiz', '2e2'),
                exact: true
              },
              {
                path: '/ch10-vla-models/',
                component: ComponentCreator('/ch10-vla-models/', '890'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/ch11-humanoid-design/',
                component: ComponentCreator('/ch11-humanoid-design/', '011'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/ch12-learning-adaptation/',
                component: ComponentCreator('/ch12-learning-adaptation/', '6ae'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/ch13-multi-robot-systems/',
                component: ComponentCreator('/ch13-multi-robot-systems/', 'ee3'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/ch14-safety-ethics/',
                component: ComponentCreator('/ch14-safety-ethics/', '878'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/ch15-capstone-project/',
                component: ComponentCreator('/ch15-capstone-project/', 'af2'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/ch15-capstone-project/exercises/chapter15-quiz',
                component: ComponentCreator('/ch15-capstone-project/exercises/chapter15-quiz', 'c25'),
                exact: true
              },
              {
                path: '/getting-started/intro',
                component: ComponentCreator('/getting-started/intro', '6e4'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/reference/api-reference',
                component: ComponentCreator('/reference/api-reference', '592'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/reference/dependencies',
                component: ComponentCreator('/reference/dependencies', '03e'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/reference/glossary',
                component: ComponentCreator('/reference/glossary', '940'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/reference/hardware-specs',
                component: ComponentCreator('/reference/hardware-specs', '7bd'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/reference/quiz-template',
                component: ComponentCreator('/reference/quiz-template', 'c01'),
                exact: true,
                sidebar: "textbookSidebar"
              },
              {
                path: '/',
                component: ComponentCreator('/', '729'),
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
