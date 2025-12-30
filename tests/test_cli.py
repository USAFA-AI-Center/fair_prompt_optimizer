# test_cli.py
"""
Tests for fair_prompt_optimizer.cli module.
"""

import json
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path

from click.testing import CliRunner

from fair_prompt_optimizer.cli import main, METRICS


class TestCLIBasics:
    """Tests for basic CLI functionality"""
    
    def test_main_help(self):
        """Test main --help"""
        runner = CliRunner()
        result = runner.invoke(main, ['--help'])
        
        assert result.exit_code == 0
        assert 'FAIR Prompt Optimizer' in result.output
    
    def test_version(self):
        """Test --version flag"""
        runner = CliRunner()
        result = runner.invoke(main, ['--version'])
        
        assert result.exit_code == 0
        assert '0.1.0' in result.output


class TestInitCommand:
    """Tests for 'fair-optimize init' command"""
    
    def test_creates_config_file(self):
        """Test that init creates agent_config.json"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            result = runner.invoke(main, ['init'])
            
            assert result.exit_code == 0
            assert Path('agent_config.json').exists()
    
    def test_creates_examples_file(self):
        """Test that init creates examples.json"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            result = runner.invoke(main, ['init'])
            
            assert result.exit_code == 0
            assert Path('examples.json').exists()
    
    def test_config_is_valid_json(self):
        """Test that created config is valid JSON"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            runner.invoke(main, ['init'])
            
            with open('agent_config.json') as f:
                config = json.load(f)
            
            assert 'version' in config
            assert 'role_definition' in config
            assert 'model' in config
            assert 'agent' in config
    
    def test_examples_is_valid_json(self):
        """Test that created examples file is valid JSON"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            runner.invoke(main, ['init'])
            
            with open('examples.json') as f:
                examples = json.load(f)
            
            assert isinstance(examples, list)
            assert len(examples) >= 1
            assert 'inputs' in examples[0]
            assert 'expected_output' in examples[0]
    
    def test_config_has_default_tool(self):
        """Test that config includes SafeCalculatorTool"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            runner.invoke(main, ['init'])
            
            with open('agent_config.json') as f:
                config = json.load(f)
            
            assert 'SafeCalculatorTool' in config['agent']['tools']
    
    def test_output_message(self):
        """Test that init shows success message"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            result = runner.invoke(main, ['init'])
            
            assert 'Created starter config' in result.output
            assert 'agent_config.json' in result.output


class TestOptimizeCommand:
    """Tests for 'fair-optimize optimize' command"""
    
    def test_requires_config(self):
        """Test that --config is required"""
        runner = CliRunner()
        
        result = runner.invoke(main, ['optimize'])
        
        assert result.exit_code != 0
        assert 'Missing option' in result.output or 'required' in result.output.lower()
    
    def test_requires_training_data(self):
        """Test that --training-data is required"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            # Create dummy config
            Path('config.json').write_text('{}')
            
            result = runner.invoke(main, ['optimize', '-c', 'config.json'])
            
            assert result.exit_code != 0
    
    def test_validates_optimizer_choice(self):
        """Test that optimizer must be bootstrap or mipro"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            Path('config.json').write_text('{}')
            Path('examples.json').write_text('[]')
            
            result = runner.invoke(main, [
                'optimize',
                '-c', 'config.json',
                '-t', 'examples.json',
                '--optimizer', 'invalid'
            ])
            
            assert result.exit_code != 0
    
    def test_validates_metric_choice(self):
        """Test that metric must be valid"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            Path('config.json').write_text('{}')
            Path('examples.json').write_text('[]')
            
            result = runner.invoke(main, [
                'optimize',
                '-c', 'config.json',
                '-t', 'examples.json',
                '--metric', 'invalid'
            ])
            
            assert result.exit_code != 0
    
    def test_mipro_requires_lm(self):
        """Test that MIPROv2 requires an LM"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            # Create minimal valid files
            runner.invoke(main, ['init'])
            
            result = runner.invoke(main, [
                'optimize',
                '-c', 'agent_config.json',
                '-t', 'examples.json',
                '--optimizer', 'mipro'
                # No --ollama-model or --openai-model
            ])
            
            assert result.exit_code != 0
            assert 'requires an LM' in result.output or 'Missing LM' in result.output
    
    def test_default_output_path(self, mock_fairlib_module):
        """Test that default output path is generated"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            runner.invoke(main, ['init'])
            
            with patch('fair_prompt_optimizer.cli.create_agent_from_config') as mock_create:
                with patch('fair_prompt_optimizer.cli.FAIRPromptOptimizer') as mock_opt:
                    mock_agent = MagicMock()
                    mock_create.return_value = mock_agent
                    
                    mock_config = MagicMock()
                    mock_config.role_definition = "Test"
                    mock_config.examples = []
                    mock_config.model.adapter = "Test"
                    mock_config.model.model_name = "test"
                    mock_config.agent.tools = []
                    mock_config.metadata = MagicMock()
                    mock_config.metadata.optimizer = "bootstrap"
                    mock_config.metadata.optimized_at = "2025-01-01"
                    
                    mock_opt.return_value.compile.return_value = mock_config
                    
                    result = runner.invoke(main, [
                        'optimize',
                        '-c', 'agent_config.json',
                        '-t', 'examples.json',
                        '--optimizer', 'bootstrap'
                    ])
                    
                    # Check that output path was derived from input
                    call_kwargs = mock_opt.return_value.compile.call_args.kwargs
                    assert 'agent_config_optimized.json' in call_kwargs.get('output_path', '')
    
    def test_verbose_flag(self):
        """Test that -v enables verbose output"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            runner.invoke(main, ['init'])
            
            with patch('fair_prompt_optimizer.cli.create_agent_from_config'):
                with patch('fair_prompt_optimizer.cli.FAIRPromptOptimizer'):
                    with patch('logging.basicConfig') as mock_logging:
                        runner.invoke(main, [
                            'optimize',
                            '-c', 'agent_config.json',
                            '-t', 'examples.json',
                            '-v'
                        ])
                        
                        # Verify DEBUG level was set
                        import logging
                        calls = mock_logging.call_args_list
                        # At least one call should have DEBUG level
                        assert any(
                            call.kwargs.get('level') == logging.DEBUG 
                            for call in calls
                        )


class TestOptimizeWithOllama:
    """Tests for optimization with Ollama"""
    
    def test_ollama_model_accepted(self, mock_fairlib_module):
        """Test that --ollama-model is accepted for mipro"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            runner.invoke(main, ['init'])
            
            with patch('fair_prompt_optimizer.cli.create_agent_from_config') as mock_create:
                with patch('fair_prompt_optimizer.cli.FAIRPromptOptimizer') as mock_opt:
                    with patch('dspy.LM') as mock_lm:
                        mock_agent = MagicMock()
                        mock_create.return_value = mock_agent
                        
                        mock_config = MagicMock()
                        mock_config.role_definition = "Test"
                        mock_config.examples = []
                        mock_config.model.adapter = "Test"
                        mock_config.model.model_name = "test"
                        mock_config.agent.tools = []
                        mock_config.metadata = MagicMock()
                        
                        mock_opt.return_value.compile.return_value = mock_config
                        
                        result = runner.invoke(main, [
                            'optimize',
                            '-c', 'agent_config.json',
                            '-t', 'examples.json',
                            '--optimizer', 'mipro',
                            '--ollama-model', 'llama3:8b'
                        ])
                        
                        # Verify DSPy LM was created with Ollama
                        mock_lm.assert_called()
                        call_args = mock_lm.call_args
                        assert 'ollama' in str(call_args).lower()


class TestOptimizeWithOpenAI:
    """Tests for optimization with OpenAI"""
    
    def test_openai_model_accepted(self, mock_fairlib_module):
        """Test that --openai-model is accepted for mipro"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            runner.invoke(main, ['init'])
            
            with patch('fair_prompt_optimizer.cli.create_agent_from_config') as mock_create:
                with patch('fair_prompt_optimizer.cli.FAIRPromptOptimizer') as mock_opt:
                    with patch('dspy.LM') as mock_lm:
                        mock_agent = MagicMock()
                        mock_create.return_value = mock_agent
                        
                        mock_config = MagicMock()
                        mock_config.role_definition = "Test"
                        mock_config.examples = []
                        mock_config.model.adapter = "Test"
                        mock_config.model.model_name = "test"
                        mock_config.agent.tools = []
                        mock_config.metadata = MagicMock()
                        
                        mock_opt.return_value.compile.return_value = mock_config
                        
                        result = runner.invoke(main, [
                            'optimize',
                            '-c', 'agent_config.json',
                            '-t', 'examples.json',
                            '--optimizer', 'mipro',
                            '--openai-model', 'gpt-4o-mini'
                        ])
                        
                        # Verify DSPy LM was created with OpenAI
                        mock_lm.assert_called()
                        call_args = mock_lm.call_args
                        assert 'openai' in str(call_args).lower()


class TestTestCommand:
    """Tests for 'fair-optimize test' command"""
    
    def test_requires_config(self):
        """Test that --config is required"""
        runner = CliRunner()
        
        result = runner.invoke(main, ['test'])
        
        assert result.exit_code != 0
    
    def test_loads_agent(self, mock_fairlib_module):
        """Test that test command loads the agent"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            runner.invoke(main, ['init'])
            
            with patch('fair_prompt_optimizer.cli.create_agent_from_config') as mock_create:
                mock_agent = MagicMock()
                mock_agent.arun = AsyncMock(return_value="Test response")
                mock_create.return_value = mock_agent
                
                # Send 'exit' immediately to quit the loop
                result = runner.invoke(
                    main, 
                    ['test', '-c', 'agent_config.json'],
                    input='exit\n'
                )
                
                assert result.exit_code == 0
                mock_create.assert_called_once()
    
    def test_interactive_session(self, mock_fairlib_module):
        """Test interactive chat session"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            runner.invoke(main, ['init'])
            
            with patch('fair_prompt_optimizer.cli.create_agent_from_config') as mock_create:
                mock_agent = MagicMock()
                mock_agent.arun = AsyncMock(return_value="42")
                mock_create.return_value = mock_agent
                
                # Send a query then exit
                result = runner.invoke(
                    main,
                    ['test', '-c', 'agent_config.json'],
                    input='What is 6*7?\nexit\n'
                )
                
                assert result.exit_code == 0
                # Agent should have been called
                mock_agent.arun.assert_called()


class TestInfoCommand:
    """Tests for 'fair-optimize info' command"""
    
    def test_shows_info(self):
        """Test that info command displays help"""
        runner = CliRunner()
        result = runner.invoke(main, ['info'])
        
        assert result.exit_code == 0
        assert 'FAIR Prompt Optimizer' in result.output
    
    def test_shows_optimizers(self):
        """Test that info shows available optimizers"""
        runner = CliRunner()
        result = runner.invoke(main, ['info'])
        
        assert 'bootstrap' in result.output.lower()
        assert 'mipro' in result.output.lower()
    
    def test_shows_metrics(self):
        """Test that info shows available metrics"""
        runner = CliRunner()
        result = runner.invoke(main, ['info'])
        
        assert 'exact' in result.output.lower()
        assert 'contains' in result.output.lower()
        assert 'numeric' in result.output.lower()
        assert 'fuzzy' in result.output.lower()
    
    def test_shows_quick_start(self):
        """Test that info shows quick start guide"""
        runner = CliRunner()
        result = runner.invoke(main, ['info'])
        
        assert 'Quick Start' in result.output or 'quick start' in result.output.lower()


class TestMetricsAvailability:
    """Tests for metric availability in CLI"""
    
    def test_all_metrics_registered(self):
        """Test that all expected metrics are registered"""
        assert 'exact' in METRICS
        assert 'contains' in METRICS
        assert 'numeric' in METRICS
        assert 'fuzzy' in METRICS
    
    def test_metrics_are_callable(self):
        """Test that all metrics are callable"""
        for name, metric in METRICS.items():
            assert callable(metric), f"Metric {name} is not callable"


class TestCLIErrorHandling:
    """Tests for CLI error handling"""
    
    def test_missing_config_file(self):
        """Test error when config file doesn't exist"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            result = runner.invoke(main, [
                'optimize',
                '-c', 'nonexistent.json',
                '-t', 'examples.json'
            ])
            
            assert result.exit_code != 0
    
    def test_missing_training_file(self):
        """Test error when training file doesn't exist"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            Path('config.json').write_text('{}')
            
            result = runner.invoke(main, [
                'optimize',
                '-c', 'config.json',
                '-t', 'nonexistent.json'
            ])
            
            assert result.exit_code != 0
    
    def test_invalid_json_config(self):
        """Test error on invalid JSON config"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            Path('config.json').write_text('not valid json')
            Path('examples.json').write_text('[]')
            
            result = runner.invoke(main, [
                'optimize',
                '-c', 'config.json',
                '-t', 'examples.json'
            ])
            
            assert result.exit_code != 0


class TestCLIIntegration:
    """Integration tests for CLI"""
    
    def test_full_workflow(self, mock_fairlib_module):
        """Test complete CLI workflow: init -> optimize"""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            result = runner.invoke(main, ['init'])
            assert result.exit_code == 0
            assert Path('agent_config.json').exists()
            assert Path('examples.json').exists()
            
            with patch('fair_prompt_optimizer.cli.create_agent_from_config') as mock_create:
                with patch('fair_prompt_optimizer.cli.FAIRPromptOptimizer') as mock_opt:
                    mock_agent = MagicMock()
                    mock_create.return_value = mock_agent
                    
                    mock_config = MagicMock()
                    mock_config.role_definition = "Optimized role"
                    mock_config.examples = ["Demo 1", "Demo 2"]
                    mock_config.model.adapter = "HuggingFaceAdapter"
                    mock_config.model.model_name = "test-model"
                    mock_config.agent.tools = ["SafeCalculatorTool"]
                    mock_config.metadata = MagicMock()
                    mock_config.metadata.optimizer = "bootstrap"
                    mock_config.metadata.optimized_at = "2025-01-01"
                    
                    mock_opt.return_value.compile.return_value = mock_config
                    
                    result = runner.invoke(main, [
                        'optimize',
                        '-c', 'agent_config.json',
                        '-t', 'examples.json',
                        '--optimizer', 'bootstrap'
                    ])
                    
                    assert result.exit_code == 0
                    assert 'Optimization Complete' in result.output or 'optimized' in result.output.lower()