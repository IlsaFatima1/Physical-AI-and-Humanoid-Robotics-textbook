// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  textbookSidebar: [
    {
      type: 'doc',
      id: 'getting-started/intro',
      label: 'Getting Started'
    },
    {
      type: 'category',
      label: 'Part I: Introduction and Fundamentals',
      items: [
        {
          type: 'doc',
          id: 'ch01-introduction/index',
          label: 'Chapter 1: Introduction to Physical AI and Humanoid Robotics'
        },
        {
          type: 'doc',
          id: 'ch02-fundamentals/index',
          label: 'Chapter 2: Fundamentals of Robotics and AI Integration'
        }
      ]
    },
    {
      type: 'category',
      label: 'Part II: Core Technologies',
      items: [
        {
          type: 'doc',
          id: 'ch03-ros2-architecture/index',
          label: 'Chapter 3: ROS 2 Architecture and Communication'
        },
        {
          type: 'doc',
          id: 'ch04-gazebo-simulation/index',
          label: 'Chapter 4: Gazebo Simulation Environment'
        },
        {
          type: 'doc',
          id: 'ch05-isaac-platform/index',
          label: 'Chapter 5: NVIDIA Isaac Platform'
        },
        {
          type: 'doc',
          id: 'ch06-urdf-xacro/index',
          label: 'Chapter 6: URDF and XACRO for Robot Modeling'
        }
      ]
    },
    {
      type: 'category',
      label: 'Part III: Perception and Control',
      items: [
        {
          type: 'doc',
          id: 'ch07-perception-systems/index',
          label: 'Chapter 7: Perception Systems and Computer Vision'
        },
        {
          type: 'doc',
          id: 'ch08-navigation/index',
          label: 'Chapter 8: Mobile Robot Navigation and Path Planning'
        },
        {
          type: 'doc',
          id: 'ch09-manipulation/index',
          label: 'Chapter 9: Manipulation and Control Systems'
        }
      ]
    },
    {
      type: 'category',
      label: 'Part IV: Advanced Topics',
      items: [
        {
          type: 'doc',
          id: 'ch10-vla-models/index',
          label: 'Chapter 10: Vision-Language-Action Models'
        },
        {
          type: 'doc',
          id: 'ch11-humanoid-design/index',
          label: 'Chapter 11: Humanoid Robot Design and Control'
        },
        {
          type: 'doc',
          id: 'ch12-learning-adaptation/index',
          label: 'Chapter 12: Learning and Adaptation in Robotics'
        },
        {
          type: 'doc',
          id: 'ch13-multi-robot-systems/index',
          label: 'Chapter 13: Multi-Robot Systems and Coordination'
        },
        {
          type: 'doc',
          id: 'ch14-safety-ethics/index',
          label: 'Chapter 14: Safety and Ethics in Robotics'
        }
      ]
    },
    {
      type: 'category',
      label: 'Part V: Capstone Project',
      items: [
        {
          type: 'doc',
          id: 'ch15-capstone-project/index',
          label: 'Chapter 15: Capstone Project - Autonomous Humanoid with VLA'
        }
      ]
    },
    {
      type: 'category',
      label: 'Reference Materials',
      items: [
        {
          type: 'doc',
          id: 'reference/glossary',
          label: 'Glossary'
        },
        {
          type: 'doc',
          id: 'reference/hardware-specs',
          label: 'Hardware Specifications'
        },
        {
          type: 'doc',
          id: 'reference/api-reference',
          label: 'API Reference'
        },
        {
          type: 'doc',
          id: 'reference/dependencies',
          label: 'Cross-Chapter Dependencies'
        },
        {
          type: 'doc',
          id: 'reference/quiz-template',
          label: 'Quiz Template'
        }
      ]
    },
    {
      type: 'category',
      label: 'Appendices',
      items: [
        {
          type: 'doc',
          id: 'appendices/appendix-a-installation',
          label: 'Appendix A: Installation Guide'
        },
        {
          type: 'doc',
          id: 'appendices/appendix-b-setup',
          label: 'Appendix B: Development Environment Setup'
        },
        {
          type: 'doc',
          id: 'appendices/appendix-c-troubleshooting',
          label: 'Appendix C: Troubleshooting Guide'
        },
        {
          type: 'doc',
          id: 'appendices/appendix-e-code-templates',
          label: 'Appendix E: Code Templates'
        },
        {
          type: 'doc',
          id: 'appendices/appendix-g-resources',
          label: 'Appendix G: Instructor Resources'
        }
      ]
    }
  ]
};

module.exports = sidebars;