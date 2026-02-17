STAGING_TABLE_SUB_SCHEMA = """
    CREATE TABLE staging_sub (
        adsh VARCHAR(20) NOT NULL,
        cik INT NOT NULL,
        name VARCHAR(255) NOT NULL,
        sic INT NULL,
        countryba CHAR(10) NULL,
        stprba CHAR(10) NULL,
        cityba VARCHAR(50) NULL,
        zipba VARCHAR(15) NULL,
        bas1 VARCHAR(100) NULL,
        bas2 VARCHAR(100) NULL,
        baph VARCHAR(30) NULL,
        countryma CHAR(2) NULL,
        stprma CHAR(10) NULL,
        cityma VARCHAR(50) NULL,
        zipma VARCHAR(15) NULL,
        mas1 VARCHAR(100) NULL,
        mas2 VARCHAR(100) NULL,
        countryinc CHAR(10) NULL,
        stprinc CHAR(10) NULL,
        ein VARCHAR(15) NULL,
        former VARCHAR(255) NULL,
        changed DATE NULL,
        afs VARCHAR(10) NULL,
        wksi BOOLEAN NULL,
        fye CHAR(10) NULL,
        form VARCHAR(32) NOT NULL,
        "period" DATE NOT NULL,
        fy INT NULL,
        fp VARCHAR(10) NULL,
        filed DATE NULL,
        accepted TIMESTAMP NULL,
        prevrpt BOOLEAN NULL,
        detail BOOLEAN NULL,
        instance VARCHAR(256) NULL,
        nciks INT NULL,
        aciks VARCHAR(1024) NULL,
        PRIMARY KEY (adsh)
    );
"""

STAGING_TABLE_NUM_SCHEMA="""
    CREATE TABLE staging_num (
        adsh VARCHAR(20),
        tag VARCHAR(256) NOT NULL,
        version VARCHAR(256) NOT NULL,
        ddate DATE NOT NULL,
        qtrs INT NOT NULL,
        uom VARCHAR(256) NOT NULL,
        segments TEXT NULL,
        coreg VARCHAR(256) NULL,
        value DECIMAL(28,6) NULL,
        footnote TEXT NULL
    );
"""

STAGING_TABLE_PRE_SCHEMA = """
    CREATE TABLE staging_pre (
        adsh VARCHAR(20),
        report INT NOT NULL,
        "line" INT NOT NULL,
        stmt VARCHAR(10) NULL,
        inpth INT NULL,
        rfile VARCHAR(32) NULL,
        tag VARCHAR(256) NOT NULL,
        version VARCHAR(256) NOT NULL,
        plabel VARCHAR(512) NULL,
        negating INT NULL
    );
"""

STAGING_TABLE_TAG_SCHEMA = """
CREATE TABLE staging_tag (
	tag VARCHAR(256) NOT NULL,
	version VARCHAR(20) NOT NULL,
	custom BOOLEAN NOT NULL,
	abstract BOOLEAN NOT NULL,
	datatype VARCHAR(100) NULL,
	iord VARCHAR(1) NULL,
	crdr VARCHAR(1) NULL,
	tlabel VARCHAR(512) NULL,
	doc TEXT NULL
);
"""



"""
CREATE TABLE pre (
    adsh VARCHAR(20) REFERENCES sub(adsh),
    report INT NOT NULL,
    "line" INT NOT NULL,
    stmt VARCHAR(10) NULL,
    inpth INT NULL,
    rfile VARCHAR(32) NULL,
    tag VARCHAR(256) NOT NULL,
    version VARCHAR(256) NOT NULL,
    plabel VARCHAR(512) NULL,
    negating INT NULL
);


CREATE TABLE tag (
    tag VARCHAR(256) NOT NULL,
    version VARCHAR(20) NOT NULL,
    custom BOOLEAN NOT NULL,
    abstract BOOLEAN NOT NULL,
    datatype VARCHAR(100),
    iord VARCHAR(1),
    crdr VARCHAR(1),
    tlabel VARCHAR(512),
    doc TEXT,
	UNIQUE(tag, version)
);

CREATE INDEX idx_tag_version ON tag(version);
CREATE INDEX idx_tag_tlabel ON tag(tlabel);
CREATE INDEX idx_tag_tag ON tag(tag);


"""


