pipeline {
    agent any 
    
    environment {
        VENV = 'venv'
    }
    
    stages {
            
        stage('Pre-clean Workspace') {
            steps {
                cleanWs()
            }
        }
        
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Setup Virtual Environment') {
            steps {
                sh ''' bash -c "
                    python3 -m venv ${VENV}
                    source ${VENV}/bin/activate
                    pip install --upgrade pip setuptools wheel
                "
                '''
            }
        }
        
        stage('Install Dependencies') {
            steps {
                sh ''' bash -c "
                    source ${VENV}/bin/activate
                    pip install .
                    pip install pytest pytest-asyncio
                "
                '''
            }
        }
        
        stage('Install fair_llm') {
            steps {
                sh ''' bash -c "
                    source ${VENV}/bin/activate
                    pip install git+ssh://git@github.com/USAFA-AI-Center/fair_llm.git
                "
                '''
            }
        }
        
        stage('Build') {
            steps {
                sh ''' bash -c "
                    source ${VENV}/bin/activate
                    python3 -m compileall -q fair_prompt_optimizer/
                "
                '''
                stash(name: 'compiled-results', includes: 'fair_prompt_optimizer/**/*.pyc*')
            }
        }
        
        stage('Run Unit Tests (Mocked)') {
            steps {
                sh ''' bash -c "
                    source ${VENV}/bin/activate
                    mkdir -p test_results
                    pytest tests/ --junitxml=test_results/unit-test-results.xml -v
                    echo 'Finished Unit Tests'
                "
                '''
            }
        }
        
        stage('Run Integration Tests') {
            steps {
                sh ''' bash -c "
                    source ${VENV}/bin/activate
                    mkdir -p test_results
                    pytest tests/integration/ --junitxml=test_results/integration-test-results.xml -v || true
                    echo 'Finished Integration Tests'
                "
                '''
            }
        }
    }
    
    post {
        always {
            junit 'test_results/*.xml'
            cleanWs()
        }
    }
}