#! /bin/bash

function CheckStatus() {
  if [ "${?}" != "0" ] ; then
    echo "ERROR: $1 failed!"
    exit 1
  fi
}

function Download() {
  echo ""
  echo "Downloading ${PACKAGE_NAME} from ${PACKAGE_URL} ..."

  wget --no-check-certificate -c ${PACKAGE_URL} -O ${PACKAGE_NAME}
  CheckStatus "Downloading"
}

function Unpack() {
  echo ""
  echo "Extracting ${PACKAGE_NAME} ..."

  tar zxvf ${PACKAGE_NAME}
  CheckStatus "Extracting"

  rm ${PACKAGE_NAME}
}

function Convert() {
  echo ""
  echo "Converting weights ..."

  THIS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
  python ${THIS_SCRIPT_DIR}/convert_weights.py
  CheckStatus "Conversion"
}

Download
Unpack
Convert 

