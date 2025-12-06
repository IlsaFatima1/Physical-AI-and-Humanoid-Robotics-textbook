# Quickstart Guide: Physical AI & Humanoid Robotics Textbook Development

## Prerequisites

Before starting the textbook development, ensure you have the following:

- Git installed (version 2.0 or higher)
- Node.js (version 18.0 or higher) and npm
- Python 3.8 or higher with pip
- ROS 2 Humble Hawksbill installed and sourced
- Gazebo Garden installed
- NVIDIA Isaac Sim (if available) or Isaac ROS packages
- A modern web browser for previewing documentation

## Environment Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Install Docusaurus Dependencies
```bash
npm install
```

### 3. Set up Python Environment
```bash
python -m venv textbook-env
source textbook-env/bin/activate  # On Windows: textbook-env\Scripts\activate
pip install -r requirements.txt  # If requirements file exists
```

### 4. Verify ROS 2 Installation
```bash
source /opt/ros/humble/setup.bash
ros2 --version
```

### 5. Verify Gazebo Installation
```bash
gazebo --version
```

## Textbook Development Workflow

### 1. Chapter Development Process
For each of the 15 chapters, follow this process:

1. **Create chapter directory**: `docs/chapters/ch{XX}-{chapter-name}/`
2. **Add main content**: Create `index.md` with required components
3. **Add code examples**: Create `examples/` directory with executable code
4. **Add exercises**: Create `exercises/` directory with quiz questions
5. **Validate content**: Ensure all constitution requirements are met
6. **Review and test**: Run code examples in simulation environment

### 2. Running the Documentation Server
```bash
npm start
```
This will start a local development server at http://localhost:3000

### 3. Building for Production
```bash
npm run build
```

### 4. Validating Code Examples
For each code example in the textbook:
1. Ensure the code runs in a ROS 2 Humble environment
2. Verify it works with the specified Gazebo/Isaac versions
3. Test that outputs match expected results
4. Document any dependencies or special setup requirements

## Using Claude Code for Textbook Generation

### 1. Generating Chapter Content
Use Claude Code agents to generate chapter content while adhering to the Physical AI & Humanoid Robotics Book Constitution:

- Ensure each chapter includes all required components
- Maintain consistent terminology throughout
- Include executable ROS/Gazebo/Isaac code examples
- Add appropriate diagrams and visual aids

### 2. Quality Assurance Process
- Technical accuracy verification by subject matter experts
- Code example testing in clean environments
- Consistency checks for terminology and formatting
- Accessibility review for visual and textual content
- Student feedback integration during beta phases

## Textbook Structure Reference

### Chapter Components
Each chapter must include:
- Introduction with learning objectives
- Technical explanation of concepts
- Diagrams and visual aids
- ROS/Gazebo/Isaac code examples
- Chapter summary
- Quiz with multiple question types

### Directory Structure
```
docs/
├── chapters/
│   ├── ch01-introduction/
│   │   ├── index.md
│   │   ├── examples/
│   │   └── exercises/
│   ├── ch02-fundamentals/
│   └── ... (ch03-ch15)
├── reference/
│   └── glossary.md
├── appendices/
└── assets/
    ├── diagrams/
    └── code-examples/
```

## Validation Checklist

Before finalizing each chapter, verify:

- [ ] All required components are present
- [ ] Code examples execute successfully
- [ ] Diagrams are clear and informative
- [ ] Quiz questions test understanding effectively
- [ ] Content aligns with constitution standards
- [ ] Cross-chapter dependencies are properly documented
- [ ] Terminology is consistent with glossary
- [ ] Content is accessible and inclusive

## Next Steps

1. Begin with Chapter 1: Introduction to Physical AI and Humanoid Robotics
2. Follow the sequential order to maintain cross-chapter dependencies
3. Generate glossary and appendices concurrently with chapter development
4. Conduct regular quality assurance reviews
5. Plan for final Docusaurus integration and deployment