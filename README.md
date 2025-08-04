# LLM to Digital Twin Description Framework (LLM2DTDF)

To validate an OML repository, you need to do:
```bash
cd oml-tools
gradlew.bat build # Do once for the repository
gradlew.bat oml-validate:run --args="-i <full_path>\data\DTOnto\catalog.xml -o report.txt"
```
