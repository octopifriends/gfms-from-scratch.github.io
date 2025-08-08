# Building Geospatial Foundation Models: Course Philosophy and Pedagogy

## Core Philosophy: From Consumption to Creation

**Traditional courses**: Teach students to be sophisticated **consumers** of existing tools  
**Our course**: Teach students to be informed **creators** of new capabilities

```
Traditional: "Here's Prithvi → Here's how to use it → Apply it to your problem"
Our Approach: "Here's the problem → Let's build a solution → Compare to Prithvi"
```

**Key Insight**: Building something teaches you to truly understand it; using something only teaches you its interface.

## Pedagogical Framework: Three-Stage Builder's Journey

### **🏗️ Stage 1: Foundation Building (Weeks 1-3)**
- Students implement attention mechanisms from scratch before using PyTorch layers
- Build complete data pipelines for geospatial data
- Assemble full architecture (~10M parameters) and validate it works

### **🚀 Stage 2: System Training (Weeks 4-7)**  
- Experience complete training lifecycle: debugging convergence, interpreting loss curves
- Compare student models against random initialization and state-of-the-art
- Develop evaluation methodologies for geospatial tasks

### **🎯 Stage 3: Real-World Application (Weeks 8-10)**
- Deploy working models with APIs and documentation
- Fine-tune for specific applications with honest performance analysis
- Present complete systems to external reviewers

## Key Instructional Innovations

### **1. Constructive Learning Through Implementation**
- **Build first, understand second**: Implement before reading theory
- **Embrace failure modes**: Students debug their own broken attention implementations
- **Theory from practice**: Mathematical concepts emerge from coding challenges

### **2. Comparative Learning at Scale**
- Week 3: Student architecture vs. Vision Transformer design choices
- Week 6: Student training vs. random baseline performance  
- Week 7: Student model vs. Prithvi on identical tasks
- Week 9: Custom fine-tuning vs. standard approaches

### **3. Authentic Assessment Through Deployment**
- **Working code as evidence**: No partial credit for "trying hard"
- **Deployable deliverables**: APIs that handle real satellite imagery queries
- **Honest performance comparison**: Students must accurately report limitations

### **4. Scaffold Complexity, Maintain Completeness**
- Small-scale but **complete** implementations at each stage
- Understand every component rather than being overwhelmed by any single piece
- Build intuition through end-to-end ownership

## Innovative Instructional Formats

### **Live Coding with Intentional Errors**
Instructor codes in real-time, makes realistic mistakes, demonstrates debugging process

### **Collaborative Architecture Decisions**  
Class debates design choices (patch size: 8×8 vs 16×16?) then implements and tests empirically

### **Performance Archaeology**
Systematic investigation of training curves to understand optimization dynamics

### **Real-Time Deployment**
Deploy models to live APIs during class with immediate user feedback

### **Research Paper Reverse Engineering**
Read Prithvi paper after building custom masked autoencoder—approach research with implementation experience

## Assessment Philosophy

### **Core Principles**
1. **Working implementations** over theoretical understanding
2. **Progressive complexity**: Each week builds on previous code
3. **Professional standards**: Documentation, APIs, error handling required
4. **Peer learning**: Collaborative debugging and code review

### **Weekly Rhythm**
- **Monday**: Submit working code from previous week
- **Wednesday**: Peer code review and debugging session  
- **Friday**: Integration testing and performance validation

## Expected Student Transformation

**Week 1 Student**: "I want to apply Prithvi to my research problem"  
**Week 10 Student**: "I understand when to build custom models vs. use Prithvi, and I can implement either approach"

## Educational Innovation Impact

### **First-of-its-Kind**
- Only course teaching **building** (not just using) geospatial foundation models
- Students graduate with deployable models and complete technical portfolios
- Skills transferable to any foundation model domain

### **Industry Readiness**
- Direct preparation for roles at Google, Microsoft, Planet, NASA
- Open-source contributions create professional visibility
- Portfolio projects demonstrate complete technical competency

This represents a new paradigm: teaching students to be **creators of the future** rather than **users of the present**.
